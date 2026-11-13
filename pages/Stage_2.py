"""Stage 2 Streamlit page for periodic unit cells."""
from dataclasses import asdict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cell_geometry import periodic_connectivity, tile_density
from homogenization import STRAIN_COMPONENTS
from projected_optimization import ProjectionSettings, ProjectedBulkOptimization, filter_radius_in_elements
from ui_shared import configure_plot_fonts, density_archive, density_plot, density_png, show_plot


def show_result(result, checks):
    settings = result["settings"]
    history = pd.DataFrame(result["history"])
    first, last = history.iloc[0], history.iloc[-1]
    a, b, c = st.columns(3)
    a.metric("等效二维体积模量", f"{last['bulk_modulus']:.6g}",
             f"{(last['bulk_modulus']/first['bulk_modulus']-1):+.1%}")
    b.metric("实际体积分数", f"{last['volume']:.2%}")
    c.metric("优化迭代次数", int(last["iteration"]))
    if result["converged"]:
        st.success(f"已确认收敛。最终 β={result['beta']}，连续三步满足设计变化、目标变化和体积误差容差。")
    elif result["status"] == "iteration_limit":
        st.warning(f"达到最大迭代次数，未确认收敛。最终 β={result['beta']} / {settings.beta_max}。")
    else:
        st.warning("更新停滞，未确认收敛。显示最后一次接受的有限元结果。")

    st.subheader("单胞与周期拼接")
    density = result["density"]
    threshold = st.slider("二值显示阈值", 0.1, 0.9, 0.5, 0.05)
    binary = density >= threshold
    count = st.radio("拼接数", (3, 5), horizontal=True, format_func=lambda n: f"{n} × {n}")
    kind = st.radio("拼接显示", ("物理密度", "二值预览"), horizontal=True)
    tiled = tile_density(density if kind == "物理密度" else binary, count)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout="constrained")
    for ax, data, title in zip(
        axes, (result["design"], density, binary, tiled),
        ("原始设计变量", "投影物理密度", "二值预览", f"{count} × {count} {kind}"),
    ):
        density_plot(ax, data, title)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), data.shape[1], data.shape[0],
                                  fill=False, ec="#547282", lw=1, clip_on=False))
    for offset in range(1, count):
        axes[3].axhline(offset*settings.ny-0.5, color="#7494a8", lw=0.6, ls=":")
        axes[3].axvline(offset*settings.nx-0.5, color="#7494a8", lw=0.6, ls=":")
    show_plot(fig)
    st.caption("等效性能由投影后的物理密度有限元计算。二值图用于形态与连通性检查。")
    components = periodic_connectivity(binary)
    both = any(c["wraps_x"] and c["wraps_y"] for c in components)
    msg = f"阈值结构有 {len(components)} 个周期连通分量，"
    msg += "存在双向贯通分量。" if both else "没有同时沿 X/Y 贯通的连通分量。"
    if len(components) == 1 and both:
        st.success(msg)
    else:
        st.warning(msg)
    with st.expander("周期连通分量"):
        st.dataframe(pd.DataFrame(components, columns=["elements", "wraps_x", "wraps_y"]), hide_index=True)
    if st.button("将物理密度送入阵列力学验证"):
        st.session_state.array_source = (result, count)
        st.switch_page("pages/Array_Validation.py")

    st.subheader("等效刚度矩阵")
    st.caption("平面应力 Voigt 顺序为 εxx、εyy、γxy，γxy 为工程剪应变。矩阵与输入 E 同单位。")
    stiffness = pd.DataFrame(result["stiffness"], index=STRAIN_COMPONENTS, columns=STRAIN_COMPONENTS)
    st.dataframe(stiffness, width="stretch")

    st.subheader("优化历程")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
    axes[0].plot(history["iteration"], history["bulk_modulus"], color="#087c92")
    axes[0].set(xlabel="迭代步", ylabel="等效二维体积模量")
    axes[1].plot(history["iteration"], history["volume_error"], color="#087c92")
    axes[1].axhline(0, color="#ba2042", ls="--", label="目标体积分数")
    axes[1].set(xlabel="迭代步", ylabel="物理体积分数偏差")
    axes[1].legend()
    axes[2].plot(history["iteration"], history["change"], color="#087c92")
    axes[2].axhline(settings.tol, color="#ba2042", ls="--")
    axes[2].set(xlabel="迭代步", ylabel="设计变量最大变化")
    for step in history.loc[history.beta.diff().fillna(0) != 0, "iteration"]:
        for ax in axes:
            ax.axvline(step, color="#7894a5", ls=":", lw=1)
    st.caption("竖虚线标记 β 切换。切换时在同一目标体积分数下重新投影并求解。")
    for ax in axes:
        ax.grid(alpha=0.25)
    show_plot(fig)
    with st.expander("逐步计算数据"):
        st.dataframe(history, width="stretch", hide_index=True)

    check_frame = pd.DataFrame(checks)
    if checks:
        error = check_frame[["bulk_relative_error", "volume_relative_error"]].max().max()
        with st.expander(f"中心差分灵敏度检查 · 最大相对误差 {error:.2e}"):
            st.caption("对内部设计变量的对称等价组做 ±10⁻⁵ 扰动，经周期滤波与投影后重新求解。对照体积模量及体积分数的解析导数。")
            st.dataframe(check_frame, width="stretch", hide_index=True)

    history_csv = history.to_csv(index=False).encode("utf-8-sig")
    stiffness_csv = stiffness.to_csv(index_label="component").encode("utf-8-sig")
    metadata = {
        "settings": asdict(settings), "kinematics": "plane stress",
        "strain_components": STRAIN_COMPONENTS, "bulk_modulus": float(last["bulk_modulus"]),
        "converged": result["converged"], "status": result["status"], "beta": result["beta"],
        "iterations": int(last["iteration"]), "visual_threshold": threshold,
        "periodic_components": components,
    }
    files = {
        "optimization_history.csv": history_csv, "homogenized_stiffness.csv": stiffness_csv,
        "unit_cell.png": density_png(density), "binary_preview.png": density_png(binary),
        "periodic_3x3.png": density_png(tile_density(density, 3)),
        "periodic_5x5.png": density_png(tile_density(density, 5)),
        "binary_3x3_preview.png": density_png(tile_density(binary, 3)),
        "binary_5x5_preview.png": density_png(tile_density(binary, 5)),
        "settings.json": json.dumps(metadata, ensure_ascii=False, indent=2),
    }
    if checks:
        files["sensitivity_check.csv"] = check_frame.to_csv(index=False).encode("utf-8-sig")
    archive = density_archive(files, {"density": density, "design": result["design"],
                                      "initial_density": result["initial_density"]})
    a, b, c = st.columns(3)
    a.download_button("迭代数据 CSV", history_csv, "optimization_history.csv", "text/csv")
    b.download_button("刚度矩阵 CSV", stiffness_csv, "homogenized_stiffness.csv", "text/csv")
    c.download_button("完整结果 ZIP", archive, "stage2_results.zip", "application/zip")


configure_plot_fonts()
st.title("Stage 2 · 周期微结构优化")
st.caption("平面应力 · 周期 Q4 有限元 · 固定体积分数")
st.markdown(
    r"目标为最大化等双轴应变下的二维体积模量 "
    r"$K^H=(C^H_{11}+C^H_{22}+2C^H_{12})/4$。"
    r"用 $\varepsilon_{xx}$、$\varepsilon_{yy}$、$\gamma_{xy}$ 三个单位应变工况求解完整的 $C^H$。"
)

with st.sidebar:
    st.header("单胞与材料")
    grid = st.selectbox("单胞网格", ("32 × 32", "64 × 64", "96 × 96", "自定义"))
    if grid == "自定义":
        nx = st.number_input("横向单元数", 8, 128, 32)
        ny = st.number_input("纵向单元数", 8, 128, 32)
    else:
        nx = ny = int(grid.split()[0])
    young = st.number_input("弹性模量 E", min_value=0.001, value=1.)
    poisson = st.slider("泊松比 ν", -0.45, 0.45, 0.30, 0.05)
    st.header("优化参数")
    volume = st.slider("目标材料体积分数", 0.15, 0.85, 0.50, 0.05)
    penal = st.slider("SIMP 惩罚指数", 1.0, 5.0, 3.0, 0.5)
    reference = st.slider("滤波半径（以短边 32 单元为基准）", 1.1, 4.0, 2.0, 0.1)
    radius = filter_radius_in_elements(nx, ny, reference)
    st.caption(f"当前滤波半径 {radius:.2f} 个单元")
    max_iter = st.slider("最大迭代次数", 10, 240, 100, 10)
    beta_max = st.select_slider("最终投影 β", options=(1, 2, 4, 8, 16), value=8)
    interval = st.number_input("每级 β 的最大更新步数", 5, 60, 20)
    eta = st.slider("Heaviside 投影阈值 η", 0.2, 0.8, 0.5, 0.05)
    st.header("几何对称")
    mirror_x = st.checkbox("X 镜像（x → Lx − x）")
    mirror_y = st.checkbox("Y 镜像（y → Ly − y）")
    rotation = st.checkbox("四重旋转对称", disabled=nx != ny) and nx == ny

settings = ProjectionSettings(nx=nx, ny=ny, young=young, poisson=poisson, volume_fraction=volume,
                              penal=penal, rmin=radius, max_iter=max_iter, beta_max=beta_max,
                              beta_interval=interval, eta=eta, mirror_x=mirror_x, mirror_y=mirror_y,
                              rotation=rotation)
run_checks = st.checkbox("运行中心差分灵敏度检查", value=True)
if st.button("开始周期微结构优化", type="primary"):
    st.session_state.pop("projected_stage2_result", None)
    progress = st.progress(0., text="求解周期单胞的初始状态")
    def update(iteration, _row):
        progress.progress(iteration/max_iter, text=f"已完成 {iteration} / {max_iter} 步")
    try:
        problem = ProjectedBulkOptimization(settings)
        result = problem.optimize(callback=update)
        if run_checks:
            progress.progress(1., text="正在比较解析梯度与中心差分")
        checks = problem.check_sensitivities(beta=result["beta"]) if run_checks else []
        st.session_state.projected_stage2_result = (result, checks)
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        st.error(f"周期单胞求解失败。{exc}")
    finally:
        progress.empty()

if "projected_stage2_result" in st.session_state:
    result, checks = st.session_state.projected_stage2_result
    if settings != result["settings"] or run_checks != bool(checks):
        st.info("下方显示上次求解的结果。重新运行可应用当前设置。")
    show_result(result, checks)

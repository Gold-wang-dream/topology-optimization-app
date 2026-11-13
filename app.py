"""Stage 1 entry point. Start with: python -m streamlit run app.py."""
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from design_domain import image_mask, polygon_mask
from boundary_editor import DRAW, LOAD, FIX, EDIT, edit_boundary
from optimizer import Settings, attached_nodes, get_boundary, optimize, validate_design_boundary
from ui_shared import configure_plot_fonts, density_archive, density_plot, density_png, show_plot


@st.cache_resource
def polygon_editor():
    return st.components.v2.component(
        "polygon_editor",
        html='<canvas aria-label="设计域绘图网格"></canvas>'
             '<div><button type="button" data-undo>撤销顶点</button> '
             '<button type="button" data-clear>清空</button> '
             '<button type="button" data-apply>应用多边形</button> <output aria-live="polite"></output></div>',
        css='canvas{display:block;width:100%;background:#f5f9fb;outline:1px solid #a6b8c5;'
            'cursor:crosshair;margin-bottom:.75rem} button{font:inherit;padding:.25rem .6rem} '
            'output{margin-left:.5rem}',
        js=Path(__file__).with_name("polygon_draw.js").read_text(encoding="utf-8"),
    )


def domain_input(nx, ny, mode, drawing):
    if mode == "黑白 PNG":
        uploaded = st.file_uploader("上传黑白 PNG（白色为设计域）", type="png")
        if uploaded is None:
            return None
        try:
            return image_mask(uploaded, nx, ny)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"无法读取设计域 PNG。{exc}") from exc
    if mode == "绘制多边形":
        saved = st.session_state.get("domain_vertices", [])
        if drawing:
            st.caption("依次点击网格添加顶点，再点击「应用多边形」。")
            component_state = st.session_state.get("draw_domain")
            if component_state is not None:
                saved = component_state.get("vertices", saved)
            editor = polygon_editor()(
                key="draw_domain", data={"nx": nx, "ny": ny, "vertices": saved},
                default={"vertices": saved}, on_vertices_change=lambda: None,
            )
            st.session_state.domain_vertices = saved = editor.vertices or []
        return polygon_mask(saved, nx, ny) if len(saved) >= 3 else None
    return np.ones((ny, nx), dtype=bool)


def show_results(result):
    history = pd.DataFrame(result["history"])
    final = history.iloc[-1]
    a, b, c = st.columns(3)
    a.metric("优化迭代次数", int(final["iteration"]))
    b.metric("最终柔度", f"{final['compliance']:.6g}")
    c.metric("最终体积分数", f"{final['volume']:.2%}")

    st.subheader("迭代密度")
    last = int(final["iteration"])
    if last:
        iteration = st.slider("查看迭代步", 0, last, last,
                              key=f"stage1_iteration_{st.session_state.stage1_run}")
    else:
        iteration = 0
    row = history.iloc[iteration]
    st.caption(f"第 {iteration} 步 · 柔度 {row['compliance']:.6g} · 体积分数 {row['volume']:.2%}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), layout="constrained")
    density_plot(axes[0], result["snapshots"][iteration], f"迭代 {iteration} 的密度")
    density_plot(axes[1], result["density"], "最终密度")
    show_plot(fig)

    st.subheader("柔度与体积分数")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), layout="constrained")
    for ax, column, label in zip(axes, ("compliance", "volume"), ("柔度", "体积分数")):
        ax.plot(history["iteration"], history[column], color="#087c92")
        ax.axvline(iteration, color="#94a3b8", lw=1)
        ax.set(xlabel="迭代步", ylabel=label)
        ax.grid(alpha=0.25)
    axes[1].axhline(result["settings"].volume_fraction, color="#ba2042", ls="--", label="体积分数上限")
    axes[1].legend()
    show_plot(fig)
    with st.expander("逐步计算数据"):
        st.dataframe(history, width="stretch", hide_index=True)

    csv = history.to_csv(index=False).encode("utf-8-sig")
    png = density_png(result["density"])
    archive = density_archive({
        "iteration_history.csv": csv, "optimized_density.png": png,
        "settings.json": json.dumps(asdict(result["settings"]), ensure_ascii=False, indent=2),
    }, {"density_history": np.stack(list(result["snapshots"].values())),
        "design_mask": result["design_mask"]})
    a, b, c = st.columns(3)
    a.download_button("迭代数据 CSV", csv, "iteration_history.csv", "text/csv")
    b.download_button("最终密度 PNG", png, "optimized_density.png", "image/png")
    c.download_button("完整结果 ZIP", archive, "stage1_results.zip", "application/zip")


def show_stage1():
    configure_plot_fonts()
    st.title("Stage 1 · 二维结构拓扑优化")
    st.caption("平面应力 · Q4 有限元 · SIMP 最小柔度")

    with st.sidebar:
        st.header("网格与材料")
        nx = st.slider("横向单元数", 20, 120, 60, 5)
        ny = st.slider("纵向单元数", 10, 60, 20, 5)
        young = st.number_input("弹性模量 E", min_value=0.001, value=1.)
        poisson = st.slider("泊松比 ν", -0.45, 0.45, 0.30, 0.05)
        st.header("优化参数")
        volume = st.slider("最大材料体积分数", 0.15, 0.85, 0.40, 0.05)
        penal = st.slider("SIMP 惩罚指数", 1.0, 5.0, 3.0, 0.5)
        radius = st.slider("过滤半径（单元）", 1.1, 6.0, 2.5, 0.1)
        max_iter = st.slider("最大迭代次数", 10, 200, 80, 10)

    st.subheader("模型设置")
    st.caption("坐标原点在左下角。Fx 向右为正，Fy 向上为正。")
    def open_polygon_canvas():
        if st.session_state.stage1_domain_mode == "绘制多边形":
            st.session_state.stage1_plot_mode = DRAW

    domain_mode = st.radio("设计域", ("矩形", "黑白 PNG", "绘制多边形"), horizontal=True,
                           key="stage1_domain_mode", on_change=open_polygon_canvas)
    mode = st.radio("图上操作", (DRAW, LOAD, FIX, EDIT), horizontal=True, key="stage1_plot_mode")
    mask = settings = None
    issues = []
    try:
        mask = domain_input(nx, ny, domain_mode, mode == DRAW)
        if mask is None:
            st.info("请上传 PNG，或在「绘制区域」模式应用多边形。")
        elif not mask.any():
            issues.append("设计域为空，请增大多边形或检查 PNG。")
        else:
            boundary = edit_boundary(mask.astype(float), attached_nodes(mask), mode, "stage1",
                                     fit_left_support=True)
            settings = Settings(nx=nx, ny=ny, young=young, poisson=poisson,
                                volume_fraction=volume, penal=penal, rmin=radius, max_iter=max_iter,
                                **boundary)
            f, free = get_boundary(settings)
            validate_design_boundary(mask, f, free, settings)
    except ValueError as exc:
        issues.append(str(exc))
    for issue in issues:
        st.warning(issue)

    if st.button("开始拓扑优化", type="primary", disabled=settings is None or bool(issues)):
        st.session_state.pop("topopt_result", None)
        progress = st.progress(0, text="正在求解有限元模型")
        def update(iteration, _density, _row):
            progress.progress(iteration/max_iter, text=f"已更新 {iteration} / {max_iter} 步")
        try:
            st.session_state.topopt_result = optimize(settings, mask, callback=update)
            st.session_state.stage1_run = st.session_state.get("stage1_run", 0) + 1
        except (ValueError, ArithmeticError) as exc:
            st.error(f"优化失败。{exc}")
        finally:
            progress.empty()

    result = st.session_state.get("topopt_result")
    if result is not None and "stage1_run" in st.session_state:
        if result["settings"] != settings or not np.array_equal(result["design_mask"], mask):
            st.info("边界或模型已修改。点击「开始拓扑优化」重新计算。")
        else:
            show_results(result)


st.set_page_config(page_title="二维 SIMP 拓扑优化", layout="wide")
st.navigation([
    st.Page(show_stage1, title="Stage 1 · 结构拓扑", default=True),
    st.Page("pages/Stage_2.py", title="Stage 2 · 周期微结构", url_path="Stage_2"),
    st.Page("pages/Array_Validation.py", title="阵列力学验证", url_path="Array_Validation"),
], position="top").run()

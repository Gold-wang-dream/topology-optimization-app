"""Static analysis of a finite tiled structure with ordinary nodal boundaries."""
from dataclasses import asdict
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from boundary_editor import DRAW, EDIT, FIX, LOAD, edit_boundary
from cell_geometry import tile_density
from optimizer import FiniteStructure, Settings, attached_nodes, get_boundary, validate_design_boundary
from ui_shared import configure_plot_fonts, density_archive, density_png, show_plot


def results(result, density, settings, count):
    u, reaction, _, compliance = result
    nx, ny = settings.nx, settings.ny
    nodal_u = u.reshape(ny+1, nx+1, 2)
    magnitude = np.linalg.norm(nodal_u, axis=2)
    a, b = st.columns(2)
    a.metric("有限阵列柔度", f"{compliance:.6g}")
    b.metric("最大节点位移", f"{magnitude.max():.6g}")
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    plot = ax.imshow(magnitude, origin="lower", extent=(-.5, nx+.5, -.5, ny+.5),
                     cmap="viridis", interpolation="nearest")
    ax.set(xlabel="节点 x", ylabel="节点 y", title="节点位移幅值")
    fig.colorbar(plot, ax=ax)
    show_plot(fig)

    f, free = get_boundary(settings)
    fixed = np.setdiff1d(np.arange(f.size), free)
    force_nodes = np.unique(np.r_[np.flatnonzero(f)//2, fixed//2])
    table = pd.DataFrame({
        "node": force_nodes, "x": force_nodes % (nx+1), "y": force_nodes // (nx+1),
        "Fx": f[2*force_nodes], "Fy": f[2*force_nodes+1],
        "Ux": u[2*force_nodes], "Uy": u[2*force_nodes+1],
        "Rx": reaction[2*force_nodes], "Ry": reaction[2*force_nodes+1],
    })
    with st.expander("载荷及约束节点的位移、反力"):
        st.dataframe(table, hide_index=True)
    archive = density_archive({
        "array_density.png": density_png(density),
        "boundary_nodes.csv": table.to_csv(index=False).encode("utf-8-sig"),
        "settings.json": json.dumps({"settings": asdict(settings), "tiles": count,
                                     "compliance": compliance}, ensure_ascii=False, indent=2),
    }, {"physical_density": density, "displacement": nodal_u,
        "residual_force": reaction.reshape(ny+1, nx+1, 2)})
    st.download_button("导出阵列验证 ZIP", archive, "array_validation.zip", "application/zip")


configure_plot_fonts()
st.title("阵列力学验证")
st.caption("对有限拼接结构施加节点力和零位移约束，求解静力位移、反力和柔度。单胞接缝共用节点，阵列外边界由下方设置决定。")
source = st.session_state.get("array_source")
if source is None:
    stored = st.session_state.get("projected_stage2_result")
    source = (stored[0], 3) if stored else None
if source is None:
    st.info("请先在 Stage 2 求解单胞，或从 Stage 2 点击「将物理密度送入阵列力学验证」。")
    st.page_link("pages/Stage_2.py", label="进入 Stage 2")
    st.stop()

cell, initial_count = source
s = cell["settings"]
count = st.radio("有限阵列大小", (3, 5), index=(0 if initial_count == 3 else 1),
                 horizontal=True, format_func=lambda n: f"{n} × {n}")
density = tile_density(cell["density"], count)
ny, nx = density.shape
st.caption(f"{nx} × {ny} 个 Q4 单元，使用 Stage 2 投影物理密度及相同的 E、ν、SIMP 指数。")
mode = st.radio("图上操作", (DRAW, LOAD, FIX, EDIT), horizontal=True,
                format_func=lambda m: "查看阵列" if m == DRAW else m)
boundary = edit_boundary(density, attached_nodes(density > 0), mode, "array",
                         seam=(s.nx, s.ny))
settings = Settings(nx=nx, ny=ny, young=s.young, poisson=s.poisson, penal=s.penal, **boundary)
issue = None
try:
    f, free = get_boundary(settings)
    validate_design_boundary(density > 0, f, free, settings)
except ValueError as exc:
    issue = str(exc)
    st.warning(issue)

if st.button("计算有限阵列", type="primary", disabled=issue is not None):
    st.session_state.pop("finite_array_result", None)
    try:
        with st.spinner("正在组装并求解有限阵列刚度矩阵"):
            result = FiniteStructure(settings).solve(density)
        st.session_state.finite_array_result = (result, settings, count, cell["density"])
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        st.error(f"阵列求解失败。{exc}")

stored = st.session_state.get("finite_array_result")
if stored is not None:
    result, previous, old_count, old_density = stored
    if settings == previous and count == old_count and old_density is cell["density"]:
        results(result, density, settings, count)
    else:
        st.info("模型或边界已修改。点击「计算有限阵列」更新结果。")

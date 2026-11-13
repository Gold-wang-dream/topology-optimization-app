"""Plotly node selection for finite Q4 models."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from optimizer import node_index

DRAW, LOAD, FIX, EDIT = "绘制区域", "添加载荷", "添加约束", "编辑删除"


def selected_nodes(points, valid):
    """Use the FEM IDs embedded in the node trace. Never snap another point."""
    ny, nx = np.array(valid.shape)-1
    nodes, invalid = set(), set()
    for point in points:
        if point.get("curve_number", point.get("curveNumber")) != 1:
            continue
        number = point.get("customdata")
        i, j = point.get("x"), point.get("y")
        if i is None or j is None or node_index(i, j, nx, ny) != number:
            raise ValueError("图上节点编号与当前网格不一致，请重新选择。")
        (nodes if valid[int(j), int(i)] else invalid).add((int(i), int(j)))
    return sorted(nodes, key=lambda ij: (ij[1], ij[0])), sorted(invalid)


def preset_boundary(nx, ny, preset, valid=None):
    if preset == "简支梁":
        return ([(nx//2, ny, 0., -1.)], [(0, 0, True, True), (nx, 0, False, True)])
    if preset == "悬臂梁":
        fixed = [(0, j, True, True) for j in range(ny+1) if valid is None or valid[j, 0]]
        return ([(nx, ny//2, 0., -1.)], fixed)
    return [], []


def boundary_figure(density, valid, loads, restraints, box, seam=None):
    ny, nx = density.shape
    j, i = np.indices(valid.shape)
    figure = go.Figure(go.Heatmap(
        x0=0.5, y0=0.5, z=density, zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        colorscale=[[0, "#f5f7fa"], [1, "#b9d8e2"]], zsmooth=False,
    ))
    scatter = go.Scatter if i.size <= 10000 else go.Scattergl
    figure.add_trace(scatter(
        x=i.ravel(), y=j.ravel(), customdata=(j*(nx+1)+i).ravel(),
        mode="markers", name="网格节点", showlegend=False,
        marker=dict(size=6 if i.size <= 10000 else 4, opacity=0.36,
                    color=np.where(valid.ravel(), "#38596b", "#bc7181")),
        selected=dict(marker=dict(color="#057c90", opacity=1, size=9)),
        unselected=dict(marker=dict(opacity=0.24)),
        hovertemplate="节点 %{customdata} · (%{x}, %{y})<extra></extra>",
    ))
    for fix_x, fix_y, symbol, label in (
        (True, True, "square-open", "固定 Ux、Uy"),
        (True, False, "triangle-right-open", "固定 Ux"),
        (False, True, "triangle-up-open", "固定 Uy"),
    ):
        nodes = [(x, y) for x, y, a, b in restraints if (a, b) == (fix_x, fix_y)]
        if nodes:
            x, y = zip(*nodes)
            figure.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name=label, hoverinfo="skip",
                marker=dict(symbol=symbol, size=11, color="#a65305", line=dict(width=2)),
            ))
    max_force = max((np.hypot(fx, fy) for _, _, fx, fy in loads), default=0.)
    if max_force:
        scale = min(nx, ny)*0.16/max_force
        for x, y, fx, fy in loads:
            if fx or fy:
                figure.add_annotation(x=x, y=y, ax=x-fx*scale, ay=y-fy*scale,
                                      axref="x", ayref="y", xref="x", yref="y",
                                      text="", showarrow=True, arrowhead=3,
                                      arrowwidth=2, arrowcolor="#b52244", captureevents=False)
    figure.add_shape(type="rect", x0=0, y0=0, x1=nx, y1=ny, line=dict(color="#526f81", width=1))
    if seam:
        sx, sy = seam
        for x in range(sx, nx, sx):
            figure.add_shape(type="line", x0=x, x1=x, y0=0, y1=ny, line=dict(color="#526f81", dash="dot"))
        for y in range(sy, ny, sy):
            figure.add_shape(type="line", x0=0, x1=nx, y0=y, y1=y, line=dict(color="#526f81", dash="dot"))
    pad = max(0.7, min(nx, ny)*0.18)
    figure.update_layout(
        height=475, margin=dict(l=45, r=12, t=15, b=40),
        clickmode="event+select", dragmode="select" if box else "pan",
        xaxis=dict(title="x / 单元", range=[-pad, nx+pad], constrain="domain"),
        yaxis=dict(title="y / 单元", range=[-pad, ny+pad], scaleanchor="x", scaleratio=1,
                   constrain="domain"),
        legend=dict(orientation="h", y=-0.17),
    )
    return figure


def edit_boundary(density, valid, mode, key, default="悬臂梁", seam=None, fit_left_support=False):
    ny, nx = density.shape
    state_key = key+"_boundary"
    state = st.session_state.get(state_key)
    support_mask = valid if fit_left_support else None
    if state is None or state["shape"] != density.shape:
        loads, restraints = preset_boundary(nx, ny, default, support_mask)
        state = {"shape": density.shape, "loads": loads, "restraints": restraints,
                 "revision": 0, "support_preset": default}
        st.session_state[state_key] = state

    if fit_left_support:
        # Sessions opened before support tracking used this exact full-edge list.
        if "support_preset" not in state:
            full_edge = preset_boundary(nx, ny, "悬臂梁")[1]
            state["support_preset"] = "悬臂梁" if state["restraints"] == full_edge else None
        if state["support_preset"] == "悬臂梁":
            fitted = preset_boundary(nx, ny, "悬臂梁", valid)[1]
            if fitted != state["restraints"]:
                state.update(restraints=fitted, revision=state["revision"]+1)

    def update(loads, restraints, support_preset=None):
        state.update(loads=loads, restraints=restraints, revision=state["revision"]+1,
                     support_preset=support_preset)
        st.rerun()

    def update_loads(loads):
        update(loads, state["restraints"], state.get("support_preset"))

    preset_col, apply_col = st.columns([3, 1], vertical_alignment="bottom")
    preset = preset_col.selectbox("边界预设", ("悬臂梁", "简支梁", "空白"), key=key+"_preset")
    if apply_col.button("应用预设", key=key+"_preset_apply"):
        st.session_state.pop(key+"_boundary_notice", None)
        update(*preset_boundary(nx, ny, preset, support_mask), support_preset=preset)

    loads, restraints = state["loads"], state["restraints"]
    if fit_left_support and state["support_preset"] == "悬臂梁":
        if len(restraints) < 2:
            st.warning("设计域与左边界接触的节点不足。请在「添加约束」中设置支撑，或修改设计域。")
        elif not valid[:, 0].all():
            st.caption(f"悬臂梁预设固定左边界上接触设计域的 {len(restraints)} 个节点。")
    if fit_left_support:
        valid_restraints = [row for row in restraints if valid[row[1], row[0]]]
        if len(valid_restraints) != len(restraints):
            st.warning(f"有 {len(restraints)-len(valid_restraints)} 个约束节点位于当前设计域外。")
            if st.button("移除域外约束", key=key+"_remove_outside"):
                update(loads, valid_restraints)
    load_action = None
    if mode == LOAD:
        load_action = st.radio("载荷操作", ("添加新载荷", "移动已有载荷"),
                               horizontal=True, key=key+"_load_action")
    box = mode == FIX and st.toggle("框选节点", key=key+"_box")
    figure = boundary_figure(density, valid, loads, restraints, box, seam)
    chart_key = f"{key}_nodes_{mode}_{box}_{load_action}_{state['revision']}"
    nodes = []
    if mode == DRAW:
        st.plotly_chart(figure, key=chart_key, config={"displaylogo": False})
    else:
        event = st.plotly_chart(
            figure, key=chart_key, on_select="rerun",
            selection_mode=("points", "box") if box else "points",
            config={"displaylogo": False, "scrollZoom": True},
        )
        try:
            nodes, invalid = selected_nodes(event.selection.points, valid)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            if invalid:
                st.warning(f"所选节点中有 {len(invalid)} 个在有效设计域外。请重新选择。")
                nodes = []
    if nodes:
        i, j = nodes[0]
        st.caption(f"已选 {len(nodes)} 个节点 · 首个节点 ({i}, {j})，FEM 编号 {node_index(i, j, nx, ny)}。")
    if mode == LOAD and load_action == "移动已有载荷":
        _move_load(loads, nodes, valid, key, state["revision"], update_loads)
    elif mode == LOAD:
        st.caption("点击一个节点，填写分量后添加。箭头长度按同图最大载荷等比例缩放。")
        a, b, c = st.columns([1, 1, 1], vertical_alignment="bottom")
        fx = a.number_input("Fx", value=0., key=key+"_fx")
        fy = b.number_input("Fy", value=-1., key=key+"_fy")
        if c.button("添加所选载荷", disabled=len(nodes) != 1 or not (fx or fy), key=key+"_add_load"):
            update_loads([*loads, (*nodes[0], fx, fy)])
    elif mode == FIX:
        st.caption("点击单个节点，或打开框选并拖动。相同节点的约束方向会合并。")
        a, b = st.columns([2, 1], vertical_alignment="bottom")
        axes = a.selectbox("零位移方向", ("Ux 和 Uy", "仅 Ux", "仅 Uy"), key=key+"_axes")
        if b.button("约束所选节点", disabled=not nodes, key=key+"_add_fix"):
            merged = {(i, j): (x, y) for i, j, x, y in restraints}
            for ij in nodes:
                x, y = merged.get(ij, (False, False))
                merged[ij] = x or axes != "仅 Uy", y or axes != "仅 Ux"
            update(loads, [(*ij, *fixed) for ij, fixed in merged.items()])
    elif mode == EDIT:
        _edit_entry(loads, restraints, nodes, valid, key, state["revision"],
                    update_loads, lambda fixed: update(loads, fixed))
    notice = st.session_state.pop(key+"_boundary_notice", None)
    if notice:
        st.success(notice)
    if loads:
        with st.expander(f"已保存载荷（{len(loads)} 条）"):
            for index, load in enumerate(loads):
                st.text(_load_label(index, load))
    return dict(point_loads=tuple(loads), restraints=tuple(restraints))


def _load_label(index, load):
    i, j, fx, fy = load
    return f"载荷 {index+1} · ({i}, {j}) · Fx={fx:g}, Fy={fy:g}"


def _move_load(loads, nodes, valid, key, revision, update_loads):
    if not loads:
        st.info("当前没有可移动的载荷。请先添加新载荷或应用边界预设。")
        return
    index = st.selectbox("要移动的载荷", range(len(loads)),
                         format_func=lambda k: _load_label(k, loads[k]),
                         key=f"{key}_move_entry_{revision}")
    source = loads[index]
    st.caption(f"当前位置 · {_load_label(index, source)}。点击图上的目标节点，或输入整数坐标。")
    i, j = _target_inputs(source[:2], nodes, valid.shape, f"{key}_move_{revision}_{index}")
    allowed = bool(valid[j, i])
    if not allowed:
        st.warning(f"目标节点 ({i}, {j}) 在有效设计域外。")
    if st.button("移动所选载荷", type="primary", key=key+"_move",
                 disabled=not allowed or (i, j) == source[:2]):
        moved = list(loads)
        moved[index] = (i, j, *source[2:])
        st.session_state[key+"_boundary_notice"] = f"载荷 {index+1} 已移至 ({i}, {j})。请重新计算。"
        update_loads(moved)


def _target_inputs(current, nodes, shape, scope, columns=None):
    ny, nx = np.array(shape)-1
    target = nodes[0] if len(nodes) == 1 else current
    c1, c2 = columns if columns is not None else st.columns(2, vertical_alignment="bottom")
    i = c1.number_input("目标节点 x", 0, int(nx), target[0], key=f"{scope}_{target}_x")
    j = c2.number_input("目标节点 y", 0, int(ny), target[1], key=f"{scope}_{target}_y")
    return i, j


def _edit_entry(loads, restraints, nodes, valid, key, revision, update_loads, update_restraints):
    choices = [("载荷", k) for k in range(len(loads))] + [("约束", k) for k in range(len(restraints))]
    if not choices:
        st.caption("当前没有可编辑的载荷或约束。")
        return

    def label(choice):
        kind, k = choice
        i, j, a, b = (loads if kind == "载荷" else restraints)[k]
        if kind == "载荷":
            return _load_label(k, (i, j, a, b))
        axes = "、".join(axis for axis, fixed in (("Ux", a), ("Uy", b)) if fixed)
        return f"约束 {k+1} · ({i}, {j}) · 固定 {axes}"

    kind, index = st.selectbox("选择要编辑的载荷或约束", choices, format_func=label,
                               key=f"{key}_entry_{revision}")
    is_load = kind == "载荷"
    update = update_loads if is_load else update_restraints
    entries = list(loads if is_load else restraints)
    i, j, a, b = entries[index]
    st.caption(f"已保存 · {label((kind, index))}。点击图上的节点或修改坐标后保存。")
    scope = f"{key}_{kind}_{index}_{revision}"
    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    i, j = _target_inputs((i, j), nodes, valid.shape, scope, (c1, c2))
    if is_load:
        a = c3.number_input("Fx", value=float(a), key=scope+"_a")
        b = c4.number_input("Fy", value=float(b), key=scope+"_b")
    else:
        a = c3.checkbox("固定 Ux", value=a, key=scope+"_a")
        b = c4.checkbox("固定 Uy", value=b, key=scope+"_b")
    if not valid[j, i]:
        st.warning(f"目标节点 ({i}, {j}) 在有效设计域外。")
    c1, c2 = st.columns(2)
    if c1.button("保存修改", disabled=not valid[j, i] or not (a or b), key=key+"_save"):
        entries[index] = (i, j, a, b)
        if not is_load:
            merged = {}
            for x, y, fix_x, fix_y in entries:
                prev_x, prev_y = merged.get((x, y), (False, False))
                merged[x, y] = prev_x or fix_x, prev_y or fix_y
            entries = [(*xy, *axes) for xy, axes in merged.items()]
        st.session_state[key+"_boundary_notice"] = f"{kind}已保存至 ({i}, {j})。请重新计算。"
        update(entries)
    if c2.button("删除所选记录", key=key+"_delete"):
        del entries[index]
        update(entries)

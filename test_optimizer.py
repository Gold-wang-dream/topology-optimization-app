from io import BytesIO
import unittest

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from design_domain import image_mask, polygon_mask
from boundary_editor import preset_boundary, selected_nodes
from optimizer import (FiniteStructure, Settings, attached_nodes, element_stiffness, get_boundary,
                       get_edofs, node_index, optimize, solve_displacements)

def test_ke_symmetry():
    k=element_stiffness(1.0,0.3)
    assert np.allclose(k,k.T,atol=1e-12)
    assert np.linalg.eigvalsh(k).min()>-1e-10

def test_optimization():
    s=Settings(nx=24,ny=8,max_iter=25,rmin=2.0)
    r=optimize(s)
    h=r['history']
    assert np.isfinite([t['compliance'] for t in h]).all()
    assert np.mean(r['density'])<=s.volume_fraction+1e-3
    assert h[-1]['compliance']<h[0]['compliance']
    assert len(r['snapshots'])>1


def test_custom_boundary_and_iteration_data():
    s = Settings(nx=12, ny=6, max_iter=5, rmin=1.5,
                 point_loads=((12, 3, 0.3, -0.5), (12, 3, 0.2, -0.5)),
                 restraints=((0, 0, True, True), (0, 6, True, False)))
    mask = polygon_mask([(0, 0), (1, 0.25), (1, 0.75), (0, 1)], s.nx, s.ny)
    result = optimize(s, mask)
    f, free = get_boundary(s)
    np.testing.assert_array_equal(f[f != 0], [0.5, -1.])
    np.testing.assert_array_equal(np.setdiff1d(np.arange(len(f)), free), [0, 1, 156])
    edofs = get_edofs(s.nx, s.ny)
    ke = element_stiffness(s.young, s.poisson)
    for row in result["history"]:
        density = result["snapshots"][row["iteration"]]
        assert not density[~mask].any()
        np.testing.assert_allclose(row["volume"], density[mask].mean())
        stiffness = (1e-6 + (1 - 1e-6)*density.ravel()**s.penal)[:, None, None] * ke
        k = coo_matrix((stiffness.ravel(), (np.repeat(edofs, 8), np.tile(edofs, (1, 8)).ravel())),
                       shape=(f.size, f.size)).tocsc()[free, :][:, free]
        np.testing.assert_allclose(row["compliance"], f[free] @ spsolve(k, f[free]), rtol=1e-8)
    assert list(result["snapshots"]) == list(range(s.max_iter+1))
    np.testing.assert_array_equal(result["density"], result["snapshots"][s.max_iter])


def test_domain_and_boundary_validation():
    png = BytesIO()
    pixels = np.array([[[255, 255, 255, 255], [0, 0, 0, 255]],
                       [[255, 255, 255, 0], [255, 255, 255, 255]]], dtype=np.uint8)
    Image.fromarray(pixels).save(png, format="PNG")
    png.seek(0)
    np.testing.assert_array_equal(image_mask(png, 2, 2), [[False, True], [True, False]])
    np.testing.assert_array_equal(polygon_mask([(0, 0), (1, 0), (0, 1)], 2, 2),
                                  [[True, True], [True, False]])
    with unittest.TestCase().assertRaisesRegex(ValueError, "约束不足"):
        get_boundary(Settings(restraints=((0, 0, True, True),)))
    with unittest.TestCase().assertRaisesRegex(ValueError, "载荷分量与"):
        get_boundary(Settings(point_loads=((0, 0, 0., -1.),)))
    with unittest.TestCase().assertRaisesRegex(ValueError, "载荷节点未接触"):
        optimize(Settings(nx=4, ny=4), polygon_mask([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], 4, 4))


def test_plot_node_mapping_and_recomputed_boundary():
    nx, ny = 12, 4
    mask = np.ones((ny, nx), dtype=bool)
    mask[:2, -2:] = False
    valid = attached_nodes(mask)
    number = node_index(nx, 2, nx, ny)
    points = [
        {"curve_number": 1, "x": nx, "y": 2, "customdata": number},
        {"curve_number": 1, "x": nx, "y": 0, "customdata": nx},
        {"curve_number": 0, "x": nx-0.5, "y": 0.5},
    ]
    assert selected_nodes(points, valid) == ([(nx, 2)], [(nx, 0)])
    with unittest.TestCase().assertRaisesRegex(ValueError, "编号"):
        selected_nodes([{**points[0], "customdata": number-1}], valid)
    edofs = get_edofs(nx, ny)
    assert 2*number+1 in edofs[2*nx-1]
    loads, fixed = preset_boundary(nx, ny, "悬臂梁")
    before = FiniteStructure(Settings(nx=nx, ny=ny, point_loads=tuple(loads), restraints=tuple(fixed)))
    loads = ((nx, 3, 0.4, -1.), (nx, 3, 0.1, -1.))
    after = FiniteStructure(Settings(nx=nx, ny=ny, point_loads=loads,
                                    restraints=((0, 0, True, True), (0, ny, True, False))))
    moved = node_index(nx, 3, nx, ny)
    np.testing.assert_array_equal(after.f[2*moved:2*moved+2], [0.5, -2.])
    assert after.f[2*number+1] == 0 and before.f[2*number+1] == -1
    assert 2*node_index(0, ny, nx, ny)+1 in after.free
    assert after.solve(np.ones((ny, nx)))[-1] > before.solve(np.ones((ny, nx)))[-1]
    with unittest.TestCase().assertRaisesRegex(ArithmeticError, "奇异"):
        solve_displacements(csr_matrix([[1., 0.], [0., 0.]]), np.ones(2), np.arange(2))


def test_cantilever_benchmark():
    result = optimize(Settings())
    first, last = result["history"][0], result["history"][-1]
    np.testing.assert_allclose([first["compliance"], last["compliance"]],
                               [1841.457051, 337.614101], rtol=2e-5)
    print(f"Cantilever compliance: {first['compliance']:.6f} -> {last['compliance']:.6f}")


if __name__=='__main__':
    test_ke_symmetry()
    test_optimization()
    test_custom_boundary_and_iteration_data()
    test_domain_and_boundary_validation()
    test_plot_node_mapping_and_recomputed_boundary()
    test_cantilever_benchmark()
    print("Stage 1 numerical tests passed")

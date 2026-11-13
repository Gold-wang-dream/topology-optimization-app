"""Benchmarks for the projected periodic problem and finite array equilibrium."""
import numpy as np

from cell_geometry import periodic_connectivity, tile_density
from homogenization import periodic_filter
from optimizer import FiniteStructure, Settings
from projected_optimization import (CircularFilter, ProjectedBulkOptimization, ProjectionSettings,
                                    filter_radius_in_elements)


def test_periodic_filter_scaling_and_symmetry():
    assert [filter_radius_in_elements(n, n, 2.) for n in (32, 64, 96)] == [2, 4, 6]
    data = np.random.default_rng(20).random((9, 9))
    f = CircularFilter(9, 9, 2.3)
    ref = periodic_filter(9, 9, 2.3)
    np.testing.assert_allclose(f.apply(data).ravel(), ref @ data.ravel(), atol=1e-14)
    problem = ProjectedBulkOptimization(ProjectionSettings(
        nx=9, ny=9, mirror_x=True, mirror_y=True, rotation=True))
    design = problem.symmetrize(data)
    for actual in (design, problem.evaluate(design, beta=8)[0]):
        for equivalent in (np.fliplr(actual), np.flipud(actual), np.rot90(actual)):
            np.testing.assert_allclose(actual, equivalent, atol=1e-14)


def test_projected_gradients_and_beta_continuation():
    s = ProjectionSettings(nx=14, ny=14, volume_fraction=0.43, beta_interval=5,
                           beta_max=8, max_iter=30, eta=0.46, rotation=True)
    problem = ProjectedBulkOptimization(s)
    checks = problem.check_sensitivities(samples=8)
    errors = [max(row[name] for row in checks) for name in ("bulk_relative_error", "volume_relative_error")]
    assert max(errors) < 1e-6
    print(f"Projected gradient errors: objective {errors[0]:.3e}, volume {errors[1]:.3e}")
    result = problem.optimize()
    hist = result["history"]
    assert {row["beta"] for row in hist} == {1, 2, 4, 8}
    assert max(abs(row["volume_error"]) for row in hist) < s.volume_tol
    for a, b in zip(hist, hist[1:]):
        if a["beta"] == b["beta"]:
            assert b["bulk_modulus"] >= a["bulk_modulus"]-1e-12*s.young
    density, response, _, _ = problem.evaluate(result["design"], result["beta"])
    np.testing.assert_allclose(density, result["density"])
    np.testing.assert_allclose(response.stiffness, result["stiffness"])
    short = ProjectedBulkOptimization(ProjectionSettings(nx=8, ny=8, max_iter=1)).optimize()
    assert short["status"] == "iteration_limit" and not short["converged"]
    # A uniform solid has an exact feasible fixed point, including at the last beta.
    solid = ProjectedBulkOptimization(ProjectionSettings(
        nx=8, ny=8, volume_fraction=1., beta_max=2, max_iter=12)).optimize()
    assert solid["converged"] and solid["beta"] == 2
    print(f"Projected bulk modulus: {hist[0]['bulk_modulus']:.6f} -> {hist[-1]['bulk_modulus']:.6f}")


def test_connectivity_and_tiling():
    material = np.zeros((6, 6), dtype=bool)
    material[2, [0, -1]] = True
    assert periodic_connectivity(material) == [{"elements": 2, "wraps_x": False, "wraps_y": False}]
    material[2, :] = True
    assert periodic_connectivity(material)[0]["wraps_x"]
    assert not periodic_connectivity(material)[0]["wraps_y"]
    material[:, 3] = True
    material[0, 0] = True
    components = periodic_connectivity(material)
    assert len(components) == 2 and components[0]["wraps_x"] and components[0]["wraps_y"]
    for n in (3, 5):
        tiled = tile_density(material, n)
        assert tiled.shape == (6*n, 6*n)
        for i in range(n):
            for j in range(n):
                np.testing.assert_array_equal(tiled[j*6:(j+1)*6, i*6:(i+1)*6], material)


def test_finite_array_uniform_tension():
    """Manufactured uniaxial solution on a tiled Q4 mesh, free transverse contraction."""
    density = tile_density(np.ones((2, 3)), 3)
    ny, nx = density.shape
    young, nu, traction = 2.4, 0.26, 0.15
    loads = tuple((nx, j, traction*(0.5 if j in (0, ny) else 1), 0.) for j in range(ny+1))
    fixed = tuple((0, j, True, j == 0) for j in range(ny+1))
    model = FiniteStructure(Settings(nx=nx, ny=ny, young=young, poisson=nu,
                                     point_loads=loads, restraints=fixed))
    u, reaction, _, compliance = model.solve(density)
    j, i = np.indices((ny+1, nx+1))
    expected = (traction/young*np.stack((i, -nu*j), axis=-1)).ravel()
    np.testing.assert_allclose(u, expected, atol=1e-12)
    np.testing.assert_allclose((reaction+model.f).reshape(-1, 2).sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(compliance, traction**2*nx*ny/young)
    np.testing.assert_allclose(compliance, model.f @ u, rtol=1e-12)
    print(f"Finite array tensile compliance: {compliance:.8f}")


if __name__ == "__main__":
    test_periodic_filter_scaling_and_symmetry()
    test_projected_gradients_and_beta_continuation()
    test_connectivity_and_tiling()
    test_finite_array_uniform_tension()
    print("Projection, symmetry, connectivity and finite array tests passed")

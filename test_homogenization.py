"""Numerical benchmarks for the periodic cell and the fixed-volume optimizer."""
import numpy as np

from homogenization import BulkOptimization, CellSettings, PeriodicCell, periodic_filter


def test_uniform_solid_and_uniform_gray():
    s = CellSettings(nx=7, ny=5, young=2.7, poisson=0.29)
    cell = PeriodicCell(s)
    nu = s.poisson
    constitutive = s.young/(1-nu**2) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    for rho in (0., 0.42, 1.):
        response = cell.evaluate(np.full((s.ny, s.nx), rho))
        multiplier = 1e-6+(1-1e-6)*rho**s.penal
        np.testing.assert_allclose(response.stiffness, multiplier*constitutive, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(response.bulk_modulus, multiplier*s.young/(2*(1-nu)), rtol=1e-10)
        np.testing.assert_allclose(response.fluctuation, 0, atol=1e-12)


def test_horizontal_laminate():
    s = CellSettings(nx=7, ny=6, young=3.2, poisson=0.27)
    rho = np.tile(np.array([0.5, 0.5, 0.5, 1., 1., 1.])[:, None], (1, s.nx))
    moduli = s.young*(1e-6 + (1-1e-6)*rho[:, 0]**s.penal)
    harmonic = 1/np.mean(1/moduli)
    nu = s.poisson
    c22 = harmonic/(1-nu**2)
    expected = np.array([[moduli.mean()+nu**2*c22, nu*c22, 0],
                         [nu*c22, c22, 0], [0, 0, harmonic/(2*(1+nu))]])
    cell = PeriodicCell(s)
    response = cell.evaluate(rho)
    np.testing.assert_allclose(response.stiffness, expected, rtol=1e-10, atol=1e-12)
    w = response.fluctuation
    assert np.max(np.abs(w)) > 0.01
    # Check total displacement jumps in all three imposed strain cases.
    j, i = np.indices((s.ny+1, s.nx+1))
    total = w.copy()
    total[..., 0, 0] += i
    total[..., 1, 1] += j
    total[..., 0, 2] += j/2
    total[..., 1, 2] += i/2
    dx = s.nx*np.array([[1, 0, 0], [0, 0, 0.5]])
    dy = s.ny*np.array([[0, 0, 0.5], [0, 1, 0]])
    np.testing.assert_allclose(total[:, -1]-total[:, 0], np.broadcast_to(dx, total[:, 0].shape), atol=1e-13)
    np.testing.assert_allclose(total[-1]-total[0], np.broadcast_to(dy, total[0].shape), atol=1e-13)
    np.testing.assert_allclose(cell.evaluate(np.roll(rho, 2, axis=0)).stiffness,
                               expected, rtol=1e-10, atol=1e-12)


def test_periodic_filter_and_gradient():
    s = CellSettings(nx=9, ny=7, rmin=2.3, volume_fraction=0.46)
    kernel = periodic_filter(s.nx, s.ny, s.rmin)
    np.testing.assert_allclose(np.asarray(kernel.sum(axis=0)), 1.)
    np.testing.assert_allclose(np.asarray(kernel.sum(axis=1)), 1.)
    assert kernel[0, s.nx-1] == kernel[0, 1] > 0
    assert kernel[0, (s.ny-1)*s.nx] == kernel[0, s.nx] > 0
    problem = BulkOptimization(s)
    design = np.random.default_rng(19).uniform(0.2, 0.8, (s.ny, s.nx))
    checks = problem.check_sensitivities(design, samples=9)
    error = max(row["relative_error"] for row in checks)
    assert error < 1e-6
    print(f"Maximum central-difference relative error: {error:.3e}")


def test_bulk_modulus_optimization():
    s = CellSettings(nx=16, ny=14, volume_fraction=0.45, max_iter=25, tol=0.001)
    problem = BulkOptimization(s)
    result = problem.optimize()
    history = result["history"]
    assert len(history) > 5
    bulk = np.array([row["bulk_modulus"] for row in history])
    assert np.min(np.diff(bulk)) >= -1e-12*s.young
    assert bulk[-1] > 1.10*bulk[0]
    np.testing.assert_allclose([row["volume"] for row in history], s.volume_fraction, atol=1e-10)
    final = problem.cell.evaluate(result["density"])
    np.testing.assert_allclose(final.stiffness, result["stiffness"])
    np.testing.assert_allclose(final.bulk_modulus, bulk[-1])
    assert np.linalg.eigvalsh(final.stiffness).min() > 0
    print(f"Fixed-volume bulk modulus: {bulk[0]:.6f} -> {bulk[-1]:.6f}")


if __name__ == "__main__":
    test_uniform_solid_and_uniform_gray()
    test_horizontal_laminate()
    test_periodic_filter_and_gradient()
    test_bulk_modulus_optimization()
    print("Stage 2 benchmarks passed")

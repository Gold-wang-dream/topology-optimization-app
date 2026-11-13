"""Periodic Q4 energy homogenization and bulk-modulus maximization."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from optimizer import element_stiffness, get_edofs

DENSITY_MIN = 0.001
VOID_MODULUS_RATIO = 1e-6
STRAIN_COMPONENTS = ("εxx", "εyy", "γxy")
DILATION = np.array([0.5, 0.5, 0.0])


@dataclass(frozen=True)
class CellSettings:
    nx: int = 32
    ny: int = 32
    volume_fraction: float = 0.5
    young: float = 1.0
    poisson: float = 0.3
    penal: float = 3.0
    rmin: float = 2.0
    max_iter: int = 80
    tol: float = 0.01

    def __post_init__(self):
        if any(not isinstance(n, (int, np.integer)) or n < 2 for n in (self.nx, self.ny)):
            raise ValueError("单胞的横纵单元数必须是大于等于 2 的整数。")
        if not DENSITY_MIN <= self.volume_fraction <= 1:
            raise ValueError("体积分数超出可行范围。")
        if not np.isfinite(self.young) or self.young <= 0 or not -1 < self.poisson < 0.5:
            raise ValueError("材料参数必须满足 E > 0 和 −1 < ν < 0.5。")
        if not np.isfinite([self.penal, self.rmin, self.tol]).all() or self.penal < 1 or self.rmin <= 0 or self.tol <= 0:
            raise ValueError("请检查惩罚指数、滤波半径和收敛容差。")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 0:
            raise ValueError("迭代次数必须是非负整数。")


@dataclass
class CellResponse:
    stiffness: np.ndarray
    bulk_modulus: float
    bulk_gradient: np.ndarray
    fluctuation: np.ndarray


class PeriodicCell:
    """Opposite nodes share fluctuation DOFs, including all four corners."""

    def __init__(self, settings: CellSettings):
        self.settings = settings
        nx, ny = settings.nx, settings.ny
        self.size = nx*ny
        j, i = np.indices((ny+1, nx+1))
        nodes = (j % ny)*nx + (i % nx)
        self.periodic_dofs = (2*nodes[:, :, None] + np.arange(2)).reshape(-1)
        self.edofs = self.periodic_dofs[get_edofs(nx, ny)]
        self.rows = np.repeat(self.edofs, 8, axis=1).ravel()
        self.cols = np.tile(self.edofs, (1, 8)).ravel()
        self.ke = element_stiffness(1.0, settings.poisson)

        # Columns are unit εxx, unit εyy and unit engineering shear γxy.
        # A cellwise translation can be omitted from the affine element field.
        x, y = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
        self.affine = np.zeros((8, 3))
        self.affine[0::2, 0] = x
        self.affine[1::2, 1] = y
        self.affine[0::2, 2] = y/2
        self.affine[1::2, 2] = x/2
        self.affine_forces = self.ke @ self.affine

    def evaluate(self, density) -> CellResponse:
        """Use physical densities to solve the three periodic equilibrium equations."""
        s = self.settings
        rho = np.asarray(density, dtype=float)
        if rho.shape != (s.ny, s.nx) or not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
            raise ValueError("物理密度必须是 ny × nx 的 [0, 1] 数组。")
        rho = rho.ravel()
        emin = s.young*VOID_MODULUS_RATIO
        modulus = emin + (s.young-emin)*rho**s.penal
        values = (modulus[:, None, None]*self.ke).ravel()
        k = coo_matrix((values, (self.rows, self.cols)), shape=(2*self.size, 2*self.size)).tocsc()
        k = (k + k.T)*0.5
        rhs = np.zeros((2*self.size, 3))
        np.add.at(rhs, self.edofs, -modulus[:, None, None]*self.affine_forces)
        # Fix one equivalence class in x and y to remove uniform translation.
        fluctuation = np.zeros_like(rhs)
        fluctuation[2:] = splu(k[2:, 2:].tocsc()).solve(rhs[2:])
        if not np.isfinite(fluctuation).all():
            raise ArithmeticError("周期单胞有限元求解失败。")

        element_displacements = self.affine + fluctuation[self.edofs]
        elemental_energy = np.einsum("eia,ij,ejb->eab", element_displacements, self.ke,
                                    element_displacements, optimize=True)
        # Area = nx * ny for unit square Q4 elements and unit thickness.
        stiffness = np.einsum("e,eab->ab", modulus, elemental_energy)/self.size
        stiffness = (stiffness+stiffness.T)*0.5
        bulk = float(DILATION @ stiffness @ DILATION)
        # Equilibrium removes the derivative of the fluctuation field from the energy.
        bulk_energy = np.einsum("a,eab,b->e", DILATION, elemental_energy, DILATION)
        gradient = (s.young-emin)*s.penal*rho**(s.penal-1)*bulk_energy/self.size
        return CellResponse(stiffness, bulk, gradient.reshape(s.ny, s.nx),
                            fluctuation[self.periodic_dofs].reshape(s.ny+1, s.nx+1, 2, 3))


def periodic_filter(nx, ny, radius):
    """Normalized cone filter with shortest distances across the cell seams."""
    j, i = np.indices((ny, nx))
    centers = np.column_stack((i.ravel()+0.5, j.ravel()+0.5))
    box = np.array([nx, ny])
    tree = cKDTree(centers, boxsize=box)
    distances = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    weights = np.maximum(radius-distances.data, 0)
    matrix = coo_matrix((weights, (distances.row, distances.col)), shape=distances.shape).tocsr()
    return matrix.multiply(1/np.asarray(matrix.sum(axis=1))).tocsr()


class BulkOptimization:
    """Fixed-volume SIMP problem with an energy objective and periodic density filter."""

    def __init__(self, settings: CellSettings):
        self.settings = settings
        self.cell = PeriodicCell(settings)
        self.filter = periodic_filter(settings.nx, settings.ny, settings.rmin)

    def initial_design(self):
        s = self.settings
        # A centered, smooth void seed breaks the uniform solution's symmetry.
        x = (np.arange(s.nx)+0.5)/s.nx
        y = (np.arange(s.ny)+0.5)/s.ny
        shape = np.cos(2*np.pi*x) + np.cos(2*np.pi*y[:, None])
        shape -= shape.mean()
        scale = max(np.max(np.abs(shape)), 1e-12)
        amplitude = 0.85*min(s.volume_fraction-DENSITY_MIN, 1-s.volume_fraction)
        return s.volume_fraction + amplitude*shape/scale

    def evaluate(self, design):
        s = self.settings
        # Guard against sub-ULP overshoots from the convex weighted sum.
        density = np.clip(self.filter @ design.ravel(), 0, 1).reshape(s.ny, s.nx)
        response = self.cell.evaluate(density)
        gradient = (self.filter.T @ response.bulk_gradient.ravel()).reshape(s.ny, s.nx)
        return density, response, gradient

    def check_sensitivities(self, design=None, step=1e-5, samples=5):
        """Central differences through the filter and a fresh FE solve on each side."""
        if not np.isfinite(step) or step <= 0 or samples < 1:
            raise ValueError("有限差分步长与样本数必须为正。")
        # Test a smooth interior field so no bound clips the finite differences.
        if design is None:
            design = 0.2 + 0.6*self.initial_design()
        design = np.asarray(design, dtype=float)
        eligible = np.flatnonzero((design.ravel() > step) & (design.ravel() < 1-step))
        if not eligible.size:
            raise ValueError("没有可进行中心差分的内部设计变量。")
        _, _, gradient = self.evaluate(design)
        choices = eligible[np.linspace(0, eligible.size-1, min(samples, eligible.size), dtype=int)]
        rows = []
        for index in choices:
            j, i = divmod(int(index), self.settings.nx)
            plus, minus = design.copy(), design.copy()
            plus[j, i] += step
            minus[j, i] -= step
            _, upper, _ = self.evaluate(plus)
            _, lower, _ = self.evaluate(minus)
            numerical = (upper.bulk_modulus-lower.bulk_modulus)/(2*step)
            analytic = float(gradient[j, i])
            rows.append({"i": i, "j": j, "analytic": analytic, "finite_difference": numerical,
                         "relative_error": abs(analytic-numerical)/max(abs(analytic), abs(numerical), 1e-14)})
        return rows

    def optimize(self, callback=None):
        s = self.settings
        design = self.initial_design()
        density, response, gradient = self.evaluate(design)
        history = [_record(0, density, response, 0.)]
        initial_density = density.copy()
        converged = s.volume_fraction in (DENSITY_MIN, 1)
        for iteration in range(1, s.max_iter+1):
            if converged:
                break
            proposal = _oc_update(design, gradient, s.volume_fraction)
            # Backtracking preserves the volume and accepts only an energy gain
            # (up to roundoff) if the full OC step overshoots.
            for _ in range(14):
                candidate_density, candidate_response, candidate_gradient = self.evaluate(proposal)
                if candidate_response.bulk_modulus >= response.bulk_modulus - 1e-12*s.young:
                    break
                proposal = 0.5*(design+proposal)
            else:
                break
            change = float(np.max(np.abs(proposal-design)))
            design = proposal
            density, response, gradient = candidate_density, candidate_response, candidate_gradient
            history.append(_record(iteration, density, response, change))
            if callback is not None:
                callback(iteration, history[-1])
            converged = change < s.tol
        return {"density": density, "initial_density": initial_density, "design": design,
                "stiffness": response.stiffness, "history": history, "settings": s,
                "converged": converged}


def _record(iteration, density, response, change):
    row = {"iteration": iteration, "bulk_modulus": response.bulk_modulus,
           "volume": float(density.mean()), "change": change}
    for i in range(3):
        for j in range(i, 3):
            row[f"C{i+1}{j+1}"] = float(response.stiffness[i, j])
    return row


def _oc_update(design, gradient, volume, volume_gradient=None, volume_of=np.mean):
    lower = np.maximum(DENSITY_MIN, design-0.2)
    upper = np.minimum(1, design+0.2)
    # The periodic filter is doubly stochastic, so mean(x) = mean(H @ x).
    sensitivity = gradient
    if volume_gradient is not None:
        sensitivity = gradient / np.maximum(volume_gradient, np.finfo(float).tiny)
    sensitivity = np.maximum(sensitivity, max(np.max(sensitivity)*1e-14, np.finfo(float).tiny))
    lo, hi = 0., float(np.max(sensitivity*(design/lower)**2))
    for _ in range(85):
        mid = (lo+hi)/2
        trial = np.clip(design*np.sqrt(sensitivity/mid), lower, upper)
        actual = volume_of(trial)
        if actual > volume:
            lo = mid
        else:
            hi = mid
        if abs(actual-volume) < 1e-12:
            break
    return trial

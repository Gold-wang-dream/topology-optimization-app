"""Self-contained educational 2-D SIMP minimum-compliance solver.

Plane-stress bilinear quadrilateral FEM, density filtering, OC update.
Reference formulation: Andreassen et al. (2011); DTU TopOpt educational codes.
Designed for a cantilever or simply supported beam, not production validation.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import warnings
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree


@dataclass
class Settings:
    nx: int = 60
    ny: int = 20
    volume_fraction: float = 0.4
    penal: float = 3.0
    rmin: float = 2.5
    max_iter: int = 80
    tol: float = 0.01
    young: float = 1.0
    poisson: float = 0.3
    load: float = 1.0
    case: str = "cantilever"
    load_pos: str = "middle"
    point_loads: tuple[tuple[int, int, float, float], ...] | None = None
    restraints: tuple[tuple[int, int, bool, bool], ...] | None = None


def element_stiffness(E: float, nu: float) -> np.ndarray:
    """Plane-stress Q4 element with 2x2 Gaussian quadrature; unit square."""
    D = E / (1 - nu * nu) * np.array([
        [1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]
    ], dtype=float)
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    ke = np.zeros((8, 8))
    a = 1.0 / np.sqrt(3.0)
    for xi in (-a, a):
        for eta in (-a, a):
            dN = np.array([
                [-(1-eta)/4, -(1-xi)/4],
                [ (1-eta)/4, -(1+xi)/4],
                [ (1+eta)/4,  (1+xi)/4],
                [-(1+eta)/4,  (1-xi)/4],
            ])
            J = coords.T @ dN
            grad = dN @ np.linalg.inv(J)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i] = grad[i, 0]
                B[1, 2*i+1] = grad[i, 1]
                B[2, 2*i] = grad[i, 1]
                B[2, 2*i+1] = grad[i, 0]
            ke += B.T @ D @ B * np.linalg.det(J)
    return ke


def get_edofs(nx: int, ny: int) -> np.ndarray:
    """Rows of elements run from bottom to top; nodes follow x fastest."""
    edofs = []
    for j in range(ny):
        for i in range(nx):
            nodes = [j*(nx+1)+i, j*(nx+1)+i+1,
                     (j+1)*(nx+1)+i+1, (j+1)*(nx+1)+i]
            edofs.append([2*n+d for n in nodes for d in (0, 1)])
    return np.array(edofs, dtype=int)


def get_filter(nx: int, ny: int, rmin: float) -> csr_matrix:
    centers = np.array([(i+0.5, j+0.5) for j in range(ny) for i in range(nx)])
    neighbors = cKDTree(centers).query_ball_point(centers, rmin)
    rows, cols, weights = [], [], []
    for row, nearby in enumerate(neighbors):
        ds = np.linalg.norm(centers[nearby] - centers[row], axis=1)
        rows.extend([row] * len(nearby))
        cols.extend(nearby)
        weights.extend((rmin - ds).clip(min=0).tolist())
    H = coo_matrix((weights, (rows, cols)), shape=(nx*ny, nx*ny)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H.multiply(1 / Hs[:, None]).tocsr()


def get_boundary(settings: Settings):
    nx, ny = settings.nx, settings.ny
    n_nodes = (nx+1)*(ny+1)
    ndof = 2*n_nodes
    f = np.zeros(ndof)
    if settings.case == "cantilever":
        fixed = np.array([d for j in range(ny+1) for d in (2*j*(nx+1), 2*j*(nx+1)+1)])
        y = {"top":ny, "middle":ny//2, "bottom":0}[settings.load_pos]
        node = y*(nx+1)+nx
        f[2*node+1] = -settings.load
    elif settings.case == "simply_supported":
        # left bottom is pinned (Ux=Uy=0); right bottom is roller (Uy=0)
        fixed = np.array([0, 1, 2*nx+1])
        node = ny*(nx+1) + nx//2
        f[2*node+1] = -settings.load
    else:
        raise ValueError("Unknown boundary condition")
    if settings.point_loads is not None:
        f.fill(0)
        for i, j, fx, fy in settings.point_loads:
            node = node_index(i, j, nx, ny)
            if not np.isfinite([fx, fy]).all():
                raise ValueError("载荷分量必须为有限数值。")
            f[2*node:2*node+2] += [fx, fy]
    if settings.restraints is not None:
        fixed = []
        for i, j, fix_x, fix_y in settings.restraints:
            node = node_index(i, j, nx, ny)
            fixed.extend(2*node + axis for axis, fix in enumerate((fix_x, fix_y)) if fix)
    fixed = np.unique(np.asarray(fixed, dtype=int))
    if not np.isfinite(f).all() or not np.any(f):
        raise ValueError("请至少设置一个非零净载荷。")
    if np.any(f[fixed]):
        raise ValueError("载荷分量与同一节点上被约束的位移方向冲突。")
    nodes, axes = fixed // 2, fixed % 2
    i, j = nodes % (nx + 1), nodes // (nx + 1)
    rigid_modes = np.column_stack((axes == 0, axes == 1, np.where(axes == 0, -j, i)))
    if len(fixed) < 3 or np.linalg.matrix_rank(rigid_modes) < 3:
        raise ValueError("约束不足，请约束两个平移和一个转动刚体模态。")
    free = np.setdiff1d(np.arange(ndof), fixed)
    return f, free


def node_index(i, j, nx, ny):
    if not (0 <= i <= nx and 0 <= j <= ny and int(i) == i and int(j) == j):
        raise ValueError(f"节点 ({i}, {j}) 超出网格或坐标不是整数。")
    return int(j)*(nx+1) + int(i)


def attached_nodes(mask):
    """A node is valid if it touches at least one active Q4 element."""
    ny, nx = mask.shape
    attached = np.zeros((ny+1, nx+1), dtype=bool)
    active = mask.astype(bool)
    for j, i in ((0, 0), (0, 1), (1, 0), (1, 1)):
        attached[j:j+ny, i:i+nx] |= active
    return attached


def validate_design_boundary(mask, f, free, settings=None):
    """Loads and restrained nodes must touch at least one design element."""
    attached = attached_nodes(mask)
    fixed = np.setdiff1d(np.arange(f.size), free)
    if np.any(~attached.ravel()[np.flatnonzero(f)//2]):
        raise ValueError("至少有一个载荷节点未接触可设计单元。")
    if np.any(~attached.ravel()[fixed//2]):
        raise ValueError("至少有一个约束节点未接触可设计单元。")
    if settings is not None and settings.point_loads is not None:
        for i, j, _, _ in settings.point_loads:
            if not attached[int(j), int(i)]:
                raise ValueError(f"载荷节点 ({i}, {j}) 未接触可设计单元。")


def solve_displacements(k, f, free):
    """Solve the reduced system and reject singular or unbalanced solutions."""
    reduced = k[free, :][:, free]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        try:
            solved = spsolve(reduced, f[free])
        except (MatrixRankWarning, RuntimeError) as exc:
            raise ArithmeticError("约束后的刚度矩阵奇异，无法计算位移。") from exc
    if not np.isfinite(solved).all():
        raise ArithmeticError("有限元位移非有限值，请检查约束与承载路径。")
    residual = reduced @ solved - f[free]
    if np.linalg.norm(residual, ord=np.inf) > 1e-5*np.linalg.norm(f[free], ord=np.inf):
        raise ArithmeticError("自由自由度的平衡残差过大，刚度矩阵可能病态。")
    u = np.zeros(f.size)
    u[free] = solved
    return u


class FiniteStructure:
    """Nonperiodic Q4 assembly shared by Stage 1 and finite array analysis."""

    def __init__(self, settings):
        self.settings = settings
        self.edofs = get_edofs(settings.nx, settings.ny)
        self.rows = np.repeat(self.edofs, 8, axis=1).ravel()
        self.cols = np.tile(self.edofs, (1, 8)).ravel()
        self.ke = element_stiffness(1., settings.poisson)
        self.f, self.free = get_boundary(settings)

    def solve(self, density):
        s = self.settings
        density = np.asarray(density, dtype=float)
        if density.size != s.nx*s.ny or not np.isfinite(density).all() or np.any((density < -1e-12) | (density > 1+1e-12)):
            raise ValueError("有限结构密度必须与网格对应，且介于 0 和 1。")
        emin = s.young*1e-6
        modulus = emin + np.clip(density.ravel(), 0., 1.)**s.penal*(s.young-emin)
        values = (modulus[:, None, None] * self.ke).ravel()
        ndof = self.f.size
        k = coo_matrix((values, (self.rows, self.cols)), (ndof, ndof)).tocsc()
        k = 0.5*(k+k.T)
        u = solve_displacements(k, self.f, self.free)
        ue = u[self.edofs]
        energy = np.einsum("ei,ij,ej->e", ue, self.ke, ue)
        return u, k @ u-self.f, energy, float(np.sum(modulus*energy))


def optimize(settings: Settings, design_mask: Optional[np.ndarray] = None,
             callback: Optional[Callable] = None):
    """Evaluate the initial design and each OC update, including the final design.

    The optional callback retains its original pre-update, one-based numbering.
    """
    nx, ny = settings.nx, settings.ny
    nele = nx * ny
    mask = np.ones(nele) if design_mask is None else np.asarray(design_mask, dtype=float).reshape(-1)
    if mask.size != nele or np.any((mask != 0) & (mask != 1)):
        raise ValueError("design_mask must be ny x nx with values 0 or 1")
    active = mask > 0.5
    if not active.any():
        raise ValueError("Design area is empty")
    nactive = active.sum()
    model = FiniteStructure(settings)
    A = get_filter(nx, ny, settings.rmin)
    if design_mask is not None:
        validate_design_boundary(mask.reshape(ny, nx), model.f, model.free, settings)
    x = np.where(active, settings.volume_fraction, 0.001)
    Emax = settings.young
    Emin = Emax * 1e-6
    history = []
    snapshots = {}
    change = 0.
    for iteration in range(settings.max_iter + 1):
        xphys = np.asarray(A @ x).reshape(-1) * mask
        _, _, ce, compliance = model.solve(xphys)
        volume = float(np.sum(xphys)/nactive)
        hist = {"iteration": iteration, "compliance": compliance,
                "volume": volume, "change": change}
        history.append(hist)
        snapshots[iteration] = xphys.reshape(ny, nx).copy()
        if iteration == settings.max_iter or (iteration > 0 and change < settings.tol):
            break
        dcphys = -settings.penal*(Emax-Emin)*xphys**(settings.penal-1) * ce
        dc = np.asarray(A.T @ (dcphys*mask)).ravel()
        dv = np.asarray(A.T @ mask).ravel()
        old = x.copy()
        lo, hi = 0., 1e15
        move = 0.2
        for _ in range(100):
            lam = (lo+hi)/2
            b = np.maximum(1e-15, -dc / np.maximum(dv, 1e-15) / max(lam, 1e-15))
            trial = np.maximum(0.001, np.maximum(x-move,
                       np.minimum(1.0, np.minimum(x+move, x*np.sqrt(b)))))
            trial[~active] = 0.001
            trial_phys = (A @ trial) * mask
            if np.sum(trial_phys) > settings.volume_fraction * nactive:
                lo = lam
            else:
                hi = lam
            if (hi-lo)/max(hi+lo, 1e-15) < 1e-5:
                break
        x = trial
        change = float(np.max(np.abs(x - old)))
        if callback is not None:
            callback(iteration+1, xphys.reshape(ny,nx),
                     {**hist, "iteration": iteration+1, "change": change})
    return {"density": snapshots[iteration].copy(), "history": history, "snapshots": snapshots,
            "settings": settings, "design_mask":mask.reshape(ny,nx)}

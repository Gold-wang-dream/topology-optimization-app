"""Heaviside continuation around the existing periodic FE and OC methods."""
from dataclasses import dataclass

import numpy as np

from cell_geometry import project_density, symmetry_permutations
from homogenization import BulkOptimization, CellSettings, DENSITY_MIN, PeriodicCell, _oc_update, _record


@dataclass(frozen=True)
class ProjectionSettings(CellSettings):
    beta_max: int = 8
    beta_interval: int = 20
    eta: float = 0.5
    mirror_x: bool = False
    mirror_y: bool = False
    rotation: bool = False
    objective_tol: float = 1e-4
    volume_tol: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if self.beta_max not in (1, 2, 4, 8, 16) or self.beta_interval < 1:
            raise ValueError("请检查 β 上限和延续间隔。")
        if not 0 < self.eta < 1 or not 0 < self.objective_tol or not 0 < self.volume_tol:
            raise ValueError("请检查投影阈值和收敛容差。")
        if self.rotation and self.nx != self.ny:
            raise ValueError("四重旋转对称需要正方形单胞。")
        if self.volume_fraction < float(project_density(np.asarray(DENSITY_MIN), 1., self.eta)[0]):
            raise ValueError("体积分数低于最小投影密度。")


def filter_radius_in_elements(nx, ny, reference_radius):
    """Scale by the shorter cell side relative to a 32-element reference."""
    return max(1.1, reference_radius*min(nx, ny)/32)


class CircularFilter:
    """The same normalized periodic cone weights, applied by FFT convolution."""

    def __init__(self, nx, ny, radius):
        i = np.arange(nx)
        j = np.arange(ny)
        distance = np.hypot(np.minimum(i, nx-i), np.minimum(j, ny-j)[:, None])
        weights = np.maximum(radius-distance, 0.)
        self.kernel = np.fft.rfft2(weights/weights.sum())
        self.shape = weights.shape

    def apply(self, values):
        # The cone kernel is symmetric, so the forward and adjoint are identical.
        return np.fft.irfft2(self.kernel*np.fft.rfft2(values), s=self.shape, axes=(0, 1))


class ProjectedBulkOptimization(BulkOptimization):
    def __init__(self, settings: ProjectionSettings):
        self.settings = settings
        self.cell = PeriodicCell(settings)
        self.filter = CircularFilter(settings.nx, settings.ny, settings.rmin)
        self.permutations = symmetry_permutations(
            settings.nx, settings.ny, settings.mirror_x, settings.mirror_y, settings.rotation)
        self.orbits = np.unique(np.min(self.permutations, axis=0), return_inverse=True)[1]
        self.orbit_sizes = np.bincount(self.orbits)

    def symmetrize(self, values):
        means = np.bincount(self.orbits, weights=values.ravel())/self.orbit_sizes
        return means[self.orbits].reshape(values.shape)

    def physical_density(self, design, beta):
        filtered = np.clip(self.filter.apply(design), 0., 1.)
        density, slope = project_density(filtered, beta, self.settings.eta)
        return np.clip(density, 0., 1.), slope

    def evaluate(self, design, beta=1):
        density, slope = self.physical_density(self.symmetrize(design), beta)
        response = self.cell.evaluate(density)
        objective_gradient = self.symmetrize(self.filter.apply(slope*response.bulk_gradient))
        volume_gradient = self.symmetrize(self.filter.apply(slope/density.size))
        return density, response, objective_gradient, volume_gradient

    def match_volume(self, design, beta):
        """Restore projected volume by a uniform bounded shift before each beta stage."""
        lo, hi = -1., 1.
        target = self.settings.volume_fraction
        if target == 1:
            return np.ones_like(design)
        for _ in range(50):
            shift = (lo+hi)/2
            candidate = np.clip(design+shift, DENSITY_MIN, 1.)
            volume = self.physical_density(candidate, beta)[0].mean()
            if abs(volume-target) <= 1e-12:
                return candidate
            if volume > target:
                hi = shift
            else:
                lo = shift
        return candidate

    def check_sensitivities(self, design=None, step=1e-5, samples=5, beta=None):
        s = self.settings
        beta = s.beta_max if beta is None else beta
        if design is None:
            design = 0.25+0.5*self.initial_design()
        design = self.symmetrize(np.asarray(design, dtype=float))
        if not np.isfinite(step) or step <= 0 or samples < 1:
            raise ValueError("有限差分步长与样本数必须为正。")
        _, _, db, dv = self.evaluate(design, beta)
        flat = design.ravel()
        eligible = np.unique(self.orbits[(flat > DENSITY_MIN+step) & (flat < 1-step)])
        if not eligible.size:
            raise ValueError("没有可进行双侧扰动的内部变量。")
        choices = eligible[np.linspace(0, eligible.size-1, min(samples, eligible.size), dtype=int)]
        rows = []
        for group in choices:
            direction = (self.orbits == group).reshape(design.shape)
            rp, bp, _, _ = self.evaluate(design+step*direction, beta)
            rm, bm, _, _ = self.evaluate(design-step*direction, beta)
            j, i = np.argwhere(direction)[0]
            row = {"i": int(i), "j": int(j), "orbit_size": int(direction.sum()), "beta": beta}
            for name, analytic, numerical in (
                ("bulk", db[direction].sum(), (bp.bulk_modulus-bm.bulk_modulus)/(2*step)),
                ("volume", dv[direction].sum(), (rp.mean()-rm.mean())/(2*step)),
            ):
                error = abs(analytic-numerical)/max(abs(analytic), abs(numerical), 1e-14)
                row.update({name+"_analytic": analytic, name+"_fd": numerical, name+"_relative_error": error})
            rows.append(row)
        return rows

    def optimize(self, callback=None):
        s = self.settings
        beta = 1
        design = self.match_volume(self.symmetrize(self.initial_design()), beta)
        density, response, db, dv = self.evaluate(design, beta)
        initial = density.copy()
        history = [_row(0, density, response, 0., beta, s, None)]
        stable, at_beta = 0, 0
        status = "iteration_limit"
        for iteration in range(1, s.max_iter+1):
            old_design, old_bulk = design, response.bulk_modulus
            switch_beta = beta < s.beta_max and (at_beta >= s.beta_interval or stable >= 3)
            if switch_beta:
                beta *= 2
                design = self.match_volume(design, beta)
                density, response, db, dv = self.evaluate(design, beta)
                at_beta, stable = 0, 0
            else:
                volume_of = lambda x: self.physical_density(x, beta)[0].mean()
                proposal = _oc_update(design, db, s.volume_fraction, dv, volume_of)
                for backtrack in range(12):
                    if backtrack:
                        proposal = self.match_volume(0.5*(design+proposal), beta)
                    candidate = self.evaluate(proposal, beta)
                    if candidate[1].bulk_modulus >= old_bulk-1e-12*s.young:
                        break
                else:
                    if beta < s.beta_max:
                        at_beta = s.beta_interval
                        continue
                    status = "stalled"
                    break
                design = proposal
                density, response, db, dv = candidate
                at_beta += 1
            change = float(np.max(np.abs(design-old_design)))
            row = _row(iteration, density, response, change, beta, s, None if switch_beta else old_bulk)
            history.append(row)
            if callback is not None:
                callback(iteration, row)
            if not switch_beta:
                within = (change <= s.tol and row["objective_change"] <= s.objective_tol
                          and abs(row["volume_error"]) <= s.volume_tol)
                stable = stable+1 if within else 0
                if beta == s.beta_max and stable >= 3:
                    status = "converged"
                    break
        return {"density": density, "initial_density": initial, "design": design,
                "stiffness": response.stiffness, "history": history, "settings": s,
                "beta": beta, "status": status, "converged": status == "converged"}


def _row(iteration, density, response, change, beta, settings, previous_bulk):
    row = _record(iteration, density, response, change)
    objective_change = (None if previous_bulk is None else
                        abs(response.bulk_modulus-previous_bulk)/max(abs(previous_bulk), 1e-14*settings.young))
    row.update(beta=beta, objective_change=objective_change,
               volume_error=row["volume"]-settings.volume_fraction)
    return row

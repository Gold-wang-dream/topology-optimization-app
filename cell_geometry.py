"""Density projection, cell symmetry and periodic material connectivity."""
from collections import deque
import numpy as np


def project_density(filtered, beta, eta=0.5):
    if beta == 0:
        return filtered, np.ones_like(filtered)
    denominator = np.tanh(beta*eta)+np.tanh(beta*(1-eta))
    t = np.tanh(beta*(filtered-eta))
    return (np.tanh(beta*eta)+t)/denominator, beta*(1-t*t)/denominator


def symmetry_permutations(nx, ny, mirror_x=False, mirror_y=False, rotation=False):
    """X/Y reflections reverse the corresponding coordinate; C4 is rotation."""
    ids = np.arange(nx*ny).reshape(ny, nx)
    rotations = [np.rot90(ids, k) for k in range(4)] if rotation else [ids]
    permutations = []
    for grid in rotations:
        for flip_x in range(1+int(mirror_x)):
            for flip_y in range(1+int(mirror_y)):
                permutations.append(grid[::(-1 if flip_y else 1), ::(-1 if flip_x else 1)].ravel())
    return np.unique(permutations, axis=0)


def periodic_connectivity(solid):
    """Detect connected components and actual winding cycles on the periodic grid."""
    solid = np.asarray(solid, dtype=bool)
    ny, nx = solid.shape
    seen = {}
    components = []
    for j, i in np.argwhere(solid):
        start = (int(j), int(i))
        if start in seen:
            continue
        seen[start] = start
        queue = deque([start])
        count, wraps_x, wraps_y = 0, False, False
        while queue:
            y, x = queue.popleft()
            count += 1
            lifted_y, lifted_x = seen[y, x]
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                neighbor = ((y+dy) % ny, (x+dx) % nx)
                if not solid[neighbor]:
                    continue
                lifted = (lifted_y+dy, lifted_x+dx)
                if neighbor not in seen:
                    seen[neighbor] = lifted
                    queue.append(neighbor)
                else:
                    old_y, old_x = seen[neighbor]
                    wraps_x |= old_x != lifted[1]
                    wraps_y |= old_y != lifted[0]
        components.append({"elements": count, "wraps_x": wraps_x, "wraps_y": wraps_y})
    return sorted(components, key=lambda item: item["elements"], reverse=True)


def tile_density(density, count):
    if count not in (3, 5):
        raise ValueError("拼接阵列应为 3×3 或 5×5。")
    return np.tile(density, (count, count))

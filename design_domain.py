"""Convert image or polygon input to a bottom-up grid of design elements."""
import numpy as np
from PIL import Image


def image_mask(uploaded, nx, ny):
    with Image.open(uploaded) as source:
        if source.format != "PNG":
            raise ValueError("请上传 PNG 文件。")
        # Composite transparency onto black so transparent pixels are outside.
        rgba = source.convert("RGBA")
        black = Image.new("RGB", rgba.size)
        black.paste(rgba, mask=rgba.getchannel("A"))
        pixels = black.convert("L").resize((nx, ny), Image.Resampling.NEAREST)
    return np.flipud(np.asarray(pixels)) >= 128


def polygon_mask(vertices, nx, ny):
    """Rasterize with the even-odd rule at element centers."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("至少需要三个多边形顶点。")
    if not np.isfinite(points).all() or np.any((points < 0) | (points > 1)):
        raise ValueError("多边形顶点必须位于绘图区域内。")
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny)[:, None] + 0.5) / ny
    inside = np.zeros((ny, nx), dtype=bool)
    on_edge = np.zeros_like(inside)
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        x0, y0 = start
        dx, dy = end - start
        if not (dx or dy):
            continue
        on_edge |= (
            np.isclose((x-x0)*dy, (y-y0)*dx, rtol=0, atol=1e-12)
            & (np.abs(x-x0-dx/2) <= abs(dx)/2+1e-12)
            & (np.abs(y-y0-dy/2) <= abs(dy)/2+1e-12)
        )
        if dy:
            inside ^= ((y0 > y) != (y0+dy > y)) & (x < x0 + dx*(y-y0)/dy)
    return inside | on_edge

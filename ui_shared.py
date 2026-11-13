"""Plotting and export helpers for both pages."""
import io
import zipfile

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import streamlit as st
from PIL import Image


def configure_plot_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC'):
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, *plt.rcParamsDefault['font.sans-serif']]
            break


def density_plot(ax, density, title):
    ax.imshow(density, origin="lower", cmap="gray_r", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()


def show_plot(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def density_png(density):
    output = io.BytesIO()
    pixels = np.rint(255 * np.flipud(density)).astype(np.uint8)
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def density_archive(files, arrays):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())
    return output.getvalue()

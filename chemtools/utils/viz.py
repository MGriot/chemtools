"""
This module contains utility functions for plotting and visualization.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys
import random
import math
import husl
from typing import List, Optional, Tuple
from scipy.signal import savgol_filter

from .io import directory_creator
from .misc import when_date


# --- Heatmap Annotation ---
def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im: The AxesImage to be labeled.
    data: Data used to annotate. If None, the image's data is used.
    valfmt: The format of the annotations inside the heatmap.
    textcolors: A pair of colors for values above and below a threshold.
    threshold: Value in data units for separating text colors.
    **kwargs: All other arguments are forwarded to the `text` calls.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


# --- Figure Saving ---
def matplotlib_savefig(
    savefig=False,
    DPI=None,
    output="output",
    name="",
    fig_format="png",
    transparent_background=False,
):
    """
    Standardizes the process of saving matplotlib figures.

    Args:
        savefig (bool): If True, save the figure. If False, show it.
        DPI (int): DPI for the saved figure.
        output (str): Output directory.
        name (str): Specific name for the figure.
        fig_format (str): Image format extension.
        transparent_background (bool): Whether to use a transparent background.
    """
    if savefig:
        directory_creator(output)
        filename = f'./{output}/{when_date()}_{name}_plot.{fig_format}'
        save_kwargs = {
            'transparent': transparent_background,
            'bbox_inches': 'tight'
        }
        if DPI is not None:
            save_kwargs['dpi'] = DPI
        plt.savefig(filename, **save_kwargs)

    plt.show()


# --- Plot Smoothing ---
def smooth_plot(x, y, window_length=5, polyorder=2):
    """Applies a Savitzky-Golay filter to data and plots the result."""
    y_smooth = savgol_filter(y, window_length, polyorder)
    plt.plot(x, y_smooth)
    plt.show()


# --- Color Palette Generation ---

def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    clipped = tuple(min(1.0, max(0.0, c)) for c in rgb)
    return matplotlib.colors.to_hex(clipped)


def _hex_to_rgb(hex_code: str) -> Tuple[float, float, float]:
    try:
        return matplotlib.colors.to_rgb(hex_code.lower())
    except ValueError:
        raise ValueError(f"Invalid hex code: {hex_code}")


def _hcl_to_hex(h: float, c: float, l: float) -> str:
    rgb = husl.husl_to_rgb(h % 360, c, l)
    return _rgb_to_hex(rgb)


def _hex_to_hcl(hex_code: str) -> Tuple[float, float, float]:
    rgb = _hex_to_rgb(hex_code)
    return husl.rgb_to_husl(*[x * 255 for x in rgb])


def _adjust_lightness(
    rgb: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = max(0.0, min(1.0, l * factor))
    return colorsys.hls_to_rgb(h, l, s)


class HarmonizedPaletteGenerator:
    """
    A class to generate harmonized color palettes using HUSL color space.
    """
    def __init__(self, n_colors: int, style: str = "husl", **kwargs):
        self.n_colors = n_colors
        self.style = style.lower()
        self.kwargs = kwargs
        self.palette: Optional[List[str]] = None
        self.max_retries = 3
        self.jitter_amount = 0.01
        self._validate_and_prepare_args()

    def _validate_and_prepare_args(self):
        if not isinstance(self.n_colors, int) or self.n_colors <= 0:
            raise ValueError("n_colors must be a positive integer")

        if self.style in ["manual", "light", "dark"] and "base_color" not in self.kwargs:
            h = random.uniform(0, 360)
            c = random.uniform(50, 100)
            l = random.uniform(30, 70)
            self.kwargs["base_color"] = _hcl_to_hex(h, c, l)

        if self.style == "husl":
            self.kwargs.setdefault("h", random.uniform(0, 360))
            self.kwargs.setdefault("s", 90)
            self.kwargs.setdefault("l", 60)

    def generate(self) -> List[str]:
        if self.palette is not None:
            return self.palette

        try:
            if self.style == "husl":
                palette = self._generate_husl()
            elif self.style == "manual":
                palette = self._generate_manual()
            elif self.style in ["light", "dark"]:
                palette = self._generate_tinted()
            else:
                raise ValueError(f"Unknown style: {self.style}")

            palette = self._ensure_uniqueness(palette)
            self.palette = palette[: self.n_colors]
            return self.palette

        except Exception as e:
            print(f"Warning: Failed to generate palette with {self.style} style: {e}")
            print("Falling back to default color generation")
            self.palette = self._generate_fallback()
            return self.palette

    def _generate_husl(self) -> List[str]:
        hues = np.linspace(0, 360, self.n_colors, endpoint=False)
        return [husl.husl_to_hex(h, self.kwargs["s"], self.kwargs["l"]) for h in hues]

    def _generate_manual(self) -> List[str]:
        base_hcl = _hex_to_hcl(self.kwargs["base_color"])
        rule = self.kwargs.get("rule", "triadic")
        hues = self._get_base_hues(base_hcl[0], rule)
        colors = []

        for i in range(self.n_colors):
            hue = hues[i % len(hues)]
            chroma = base_hcl[1] * (0.9 + 0.2 * (i % 2))
            lightness = base_hcl[2] * (0.8 + 0.4 * ((i // len(hues)) % 2))

            hue_jitter = random.uniform(-2, 2)
            chroma_jitter = random.uniform(-5, 5)
            lightness_jitter = random.uniform(-3, 3)

            color = (
                (hue + hue_jitter) % 360,
                max(0, min(100, chroma + chroma_jitter)),
                max(0, min(100, lightness + lightness_jitter)),
            )
            colors.append(_hcl_to_hex(*color))

        return colors

    def _generate_tinted(self) -> List[str]:
        base_rgb = _hex_to_rgb(self.kwargs["base_color"])
        return [
            _rgb_to_hex(_adjust_lightness(base_rgb, 0.5 + i * 0.25))
            for i in range(self.n_colors)
        ]

    def _generate_fallback(self) -> List[str]:
        hues = np.linspace(0, 360, self.n_colors, endpoint=False)
        return [_hcl_to_hex(h, 70, 60) for h in hues]

    def _ensure_uniqueness(self, palette: List[str]) -> List[str]:
        if len(set(palette)) == len(palette):
            return palette

        for _ in range(self.max_retries):
            seen = {}
            new_palette = []
            for color in palette:
                if color not in seen:
                    seen[color] = 1
                    new_palette.append(color)
                else:
                    h, c, l = _hex_to_hcl(color)
                    jittered = (
                        (h + random.uniform(-5, 5)) % 360,
                        max(0, min(100, c + random.uniform(-10, 10))),
                        max(0, min(100, l + random.uniform(-5, 5))),
                    )
                    new_palette.append(_hcl_to_hex(*jittered))
            palette = new_palette
            if len(set(palette)) == len(palette):
                break
        return palette

    def _get_base_hues(self, base_hue: float, rule: str) -> List[float]:
        rules = {
            "complementary": [base_hue, (base_hue + 180) % 360],
            "triadic": [base_hue, (base_hue + 120) % 360, (base_hue + 240) % 360],
            "analogous": [base_hue - 30, base_hue, base_hue + 30],
            "tetradic": [base_hue, base_hue + 90, base_hue + 180, base_hue + 270],
            "split_complementary": [base_hue, base_hue + 150, base_hue + 210],
        }
        return [h % 360 for h in rules.get(rule, [base_hue])]

    def plot(self):
        """Generates and displays a plot of the palette."""
        palette = self.generate()
        n = len(palette)
        fig, ax = plt.subplots(1, 1, figsize=(max(6, n * 1.0), 2.5))
        image_data = np.array([_hex_to_rgb(c) for c in palette]).reshape(1, n, 3)
        ax.imshow(image_data, aspect="auto")

        for i, color_hex in enumerate(palette):
            ax.text(
                i, 1.05, color_hex, ha="center", va="bottom",
                color="black", fontsize=9, transform=ax.transData,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

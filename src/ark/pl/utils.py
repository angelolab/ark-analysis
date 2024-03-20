from collections.abc import Iterable
from typing import Literal

import scanpy as sc
from matplotlib import plt, ticker


def remove_x_axis_ticks(ax: plt.Axes):
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def remove_y_axis_ticks(ax: plt.Axes):
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def set_locator_formatter(ax: plt.Axes, axis: Literal["x", "y", "xy", "yx"]) -> None:
    match axis:
        case "x":
            remove_x_axis_ticks(ax)
        case "y":
            remove_y_axis_ticks(ax)
        case "xy" | "yx":
            remove_x_axis_ticks(ax)
            remove_y_axis_ticks(ax)
        case _:
            raise ValueError("axis must be 'x', 'y' or 'xy' or 'yx'")


def remove_ticks(
    f: plt.Figure | plt.Axes | Iterable[plt.Axes] | Iterable[plt.Figure] | sc.pl.DotPlot | sc.pl.StackedViolin,
    axis: Literal["x", "y", "xy", "yx"],
) -> None:
    match f:
        case plt.Figure():
            axes = f.axes
            (set_locator_formatter(a, axis) for a in axes)
        case plt.Axes():
            set_locator_formatter(f, axis)
        case sc.pl.DotPlot() | sc.pl.StackedViolin() | sc.pl.MatrixPlot():
            axes = f.get_axes()
            (remove_ticks(f=a, axis=axis) for a in axes)
        case list() | Iterable():
            (remove_ticks(f=a, axis=axis) for a in f)
        case _:
            raise ValueError("f must be a DotPlot, StackedViolin, Axes object, Figure object, or an iterable of them.")

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from theme import BAR_BASE, BAR_FOCUS, BAR_SINGLE, BLUE_RAMP, TOKENS

INK = TOKENS["ink"]
INK_SECONDARY = TOKENS["ink-secondary"]
INK_MUTED = TOKENS["ink-muted"]
SURFACE = TOKENS["surface"]
HAIRLINE = TOKENS["hairline"]
HEATMAP = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
LIGHT_TEXT_ABOVE = 0.55


def pretty(label):
    return label.replace("_", " ")


def new_figure(width, height):
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=10.5)
    return fig, ax


def confusion_figure(y_true, y_pred, classes, as_percent):
    counts = confusion_matrix(y_true, y_pred, labels=classes)
    shares = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    shown = shares if as_percent else counts

    fig, ax = new_figure(5.4, 4.8)
    ax.spines["bottom"].set_visible(False)
    ax.imshow(shares, cmap=HEATMAP, vmin=0, vmax=1)
    labels = [pretty(c) for c in classes]
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks, labels, rotation=35, ha="right")
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Predicted", color=INK_SECONDARY, fontsize=11, labelpad=8)
    ax.set_ylabel("Actual", color=INK_SECONDARY, fontsize=11, labelpad=8)

    for i in ticks:
        for j in ticks:
            text = f"{shown[i, j]:.0%}" if as_percent else f"{counts[i, j]}"
            if shown[i, j] == 0:
                color = INK_MUTED
            else:
                color = "white" if shares[i, j] > LIGHT_TEXT_ABOVE else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=10.5, color=color)

    ax.set_xticks(ticks - 0.5, minor=True)
    ax.set_yticks(ticks - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    return fig


def _bar_labels(ax, bars, values, fmt):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            fmt(value),
            va="center",
            ha="left",
            fontsize=11,
            color=INK,
        )


def metric_bars_figure(table, metric, focus):
    ordered = table[metric].sort_values()
    fig, ax = new_figure(5.2, 3.1)
    colors = [BAR_FOCUS if name == focus else BAR_BASE for name in ordered.index]
    bars = ax.barh(ordered.index, ordered.values, height=0.52, color=colors)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.grid(True, color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11, colors=INK)
    _bar_labels(ax, bars, ordered.values, lambda v: f"{v:.4f}")
    fig.tight_layout()
    return fig


def class_counts_figure(labels, counts):
    order = np.argsort(counts)
    names = [pretty(labels[i]) for i in order]
    values = np.asarray(counts)[order]
    fig, ax = new_figure(5.2, 3.3)
    bars = ax.barh(names, values, height=0.55, color=BAR_SINGLE)
    ax.set_xlim(0, values.max() * 1.15)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11, colors=INK)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + values.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=11,
            color=INK,
        )
    fig.tight_layout()
    return fig

"""Publication figures for final norms, layer tails, attention, and PCA."""

from pathlib import Path
from typing import Any
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .config import Config
from .models import preprocess
from .norms import analyze


def save_figure(fig: Figure, results: Path, config: Config, name: str, pdf: bool = True) -> None:
    """Save publication PNGs at the configured dpi and companion PDFs."""
    fig.savefig(
        results / f"figures/{name}.png",
        dpi=config["figure_dpi"],
        bbox_inches="tight",
        facecolor="white",
    )
    if pdf:
        fig.savefig(results / f"figures/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _alias(results: Path, source: str, target: str) -> None:
    """Retain old normalized figure URLs with explicitly labeled new content."""
    import shutil

    for extension in ["png", "pdf"]:
        shutil.copyfile(
            results / f"figures/{source}.{extension}", results / f"figures/{target}.{extension}"
        )


def make_figures(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> None:
    """Render A/B/C for every size in both explicitly named feature spaces."""
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable
    from .spaces import SPACES, reference_threshold

    _, layers, _ = analyze(config, sources, results)
    grid = config["image_size"] // config["patch_size"]
    selected = config["selected_indices"]
    colors = ["#cd593a", "#276987"]
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for size in config["sizes"]:
        pair = [np.load(results / f"data/{size}_{s}.npz") for s in ["no_registers", "registers"]]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
        for ax, (space, desc) in zip(axes, SPACES.items()):
            for suffix, color, label in zip(
                ["no_registers", "registers"], colors, ["Without registers", "With registers"]
            ):
                rows = [
                    r
                    for r in layers
                    if r["model"] == f"{size}_{suffix}" and r["feature_space"] == space
                ]
                ax.plot(
                    [r["layer"] for r in rows],
                    [r["high_percent"] for r in rows],
                    marker="o",
                    ms=2,
                    color=color,
                    label=label,
                )
            ax.set(
                title=desc["label"],
                xlabel="Block (0 = embedding)",
                ylabel="Patches above own layer mean + 3 std (%)",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle(f"DINOv2 {size} · raw and normalized layer tails · k={config['layer_k']:g}")
        save_figure(fig, results, config, f"high_norm_across_layers_{size}")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
        for ax, (space, desc) in zip(axes, SPACES.items()):
            values = [d[desc["norms"]].ravel() for d in pair]
            bins = np.linspace(0, max(v.max() for v in values) * 1.03, 121)
            for v, color, label in zip(values, colors, ["Without registers", "With registers"]):
                ax.hist(v, bins=bins, histtype="step", color=color, label=label)
            ax.set(title=desc["label"], xlabel="Patch L2 norm", ylabel="Patch count", yscale="log")
            ax.legend(fontsize=8)
        fig.suptitle(f"DINOv2 {size} · final feature-space diagnostic")
        save_figure(fig, results, config, f"layernorm_diagnostic_{size}")
        for space, desc in SPACES.items():
            norms = [d[desc["norms"]] for d in pair]
            cutoff = reference_threshold(config, space)
            maximum = max(v.max() for v in norms)
            bins = np.linspace(0, maximum * 1.03, config["histogram_bins"] + 1)
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, layout="constrained")
            for ax, n, color, label in zip(
                axes, norms, colors, ["Without registers", "With 4 registers"]
            ):
                ax.hist(n.ravel(), bins=bins, color=color)
                ax.axvline(cutoff, c="black", ls="--", lw=1, label=f"Reference {cutoff:.1f}")
                ax.set(
                    title=label,
                    xlabel=f"Patch L2 norm · {space}",
                    yscale="log",
                    xlim=(0, max(maximum * 1.03, cutoff * 1.05)),
                    ylim=(0.7, None),
                )
                ax.text(
                    0.97,
                    0.96,
                    f"mean {n.mean():.1f} · max {n.max():.1f}\nabove reference {(n > cutoff).mean() * 100:.3f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
                from .norms import choose_lower_threshold, choose_threshold

                gap = choose_threshold(n, config)
                lower = choose_lower_threshold(n, config)
                if gap["clear_gap"] and not np.isclose(gap["threshold"], cutoff):
                    ax.axvline(
                        gap["threshold"],
                        c="#228833",
                        ls=":",
                        label=f"High-gap split {gap['threshold']:.1f}",
                    )
                if lower["clear_gap"]:
                    ax.axvline(
                        lower["threshold"],
                        c="#8855aa",
                        ls=":",
                        label=f"Low-gap split {lower['threshold']:.1f}",
                    )
                ax.legend(fontsize=8, loc="center right")
            axes[0].set_ylabel("Patch count (log)")
            fig.suptitle(
                f"DINOv2 {size} · {desc['label']}\nReference exceedance is not a universal artifact definition",
                fontsize=11,
            )
            name = f"token_norm_histogram_{size}_{space}"
            save_figure(fig, results, config, name)
            if space == "normalized":
                _alias(results, name, f"token_norm_histogram_{size}")
            scale = Normalize(0, max(n[selected].max() for n in norms))
            attention_scale = LogNorm(
                config["attention_color_min"], max(d["cls_attention"].max() for d in pair)
            )
            fig, axes = plt.subplots(6, 7, figsize=(17.5, 15), layout="constrained")
            titles = [
                "Input",
                "Norm · no registers",
                "Norm · registers",
                "CLS attention · no reg.",
                "CLS attention · registers",
                "PCA · no registers",
                "PCA · registers",
            ]
            for r, index in enumerate(selected):
                image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
                axes[r, 0].imshow(image)
                for j, d in enumerate(pair):
                    axes[r, 1 + j].imshow(
                        norms[j][index].reshape(grid, grid),
                        cmap="magma",
                        norm=scale,
                        interpolation="nearest",
                    )
                    axes[r, 3 + j].imshow(
                        d["cls_attention"][r],
                        cmap="viridis",
                        norm=attention_scale,
                        interpolation="nearest",
                    )
                    axes[r, 5 + j].imshow(d[desc["pca"]][r], interpolation="nearest")
                for c, ax in enumerate(axes[r]):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if r == 0:
                        ax.set_title(titles[c], fontsize=9)
            fig.colorbar(
                ScalarMappable(norm=scale, cmap="magma"),
                ax=axes[:, 1:3].ravel().tolist(),
                shrink=0.25,
                label=f"Patch L2 · {space}",
            )
            fig.colorbar(
                ScalarMappable(norm=attention_scale, cmap="viridis"),
                ax=axes[:, 3:5].ravel().tolist(),
                shrink=0.25,
                label="Attention probability (log; feature-space independent)",
            )
            fig.suptitle(
                f"DINOv2 {size} · {desc['label']} for norms and PCA\nShared paired scales; PCA colors fitted independently; attention weights unchanged by final output normalization",
                fontsize=11,
            )
            name = f"visual_comparison_{size}_{space}"
            save_figure(fig, results, config, name)
            if space == "normalized":
                _alias(results, name, f"visual_comparison_{size}")
            fig, axes = plt.subplots(6, 3, figsize=(8.4, 16.2), layout="constrained")
            for r, index in enumerate(selected):
                image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
                axes[r, 0].imshow(image)
                for j, n in enumerate(norms):
                    field = n[index].reshape(grid, grid)
                    axes[r, j + 1].imshow(field, cmap="magma", norm=scale, interpolation="nearest")
                    yy, xx = np.where(field > cutoff)
                    axes[r, 0].scatter(
                        (xx + 0.5) * config["patch_size"],
                        (yy + 0.5) * config["patch_size"],
                        s=12,
                        facecolors="none",
                        edgecolors=colors[j],
                        linewidths=0.7,
                    )
                    axes[r, j + 1].set_title(
                        f"{(field > cutoff).sum()} above {cutoff:.1f}", fontsize=8
                    )
                for ax in axes[r]:
                    ax.set_xticks([])
                    ax.set_yticks([])
            for ax, title in zip(
                axes[0], ["Input + reference exceedances", "Without registers", "With registers"]
            ):
                ax.set_title(title, fontsize=9)
            fig.colorbar(
                ScalarMappable(norm=scale, cmap="magma"),
                ax=axes[:, 1:].ravel().tolist(),
                shrink=0.3,
                label=f"L2 · {space}",
            )
            fig.suptitle(
                f"DINOv2 {size} · {desc['label']}\nOrange: no registers; blue: registers; reference is a magnitude diagnostic",
                fontsize=10,
            )
            name = f"high_norm_locations_{size}_{space}"
            save_figure(fig, results, config, name)
            if space == "normalized":
                _alias(results, name, f"high_norm_locations_{size}")
            fig, axes = plt.subplots(6, 5, figsize=(12.5, 15), layout="constrained")
            roll_scale = LogNorm(
                config["attention_color_min"], max(d["rollout"].max() for d in pair)
            )
            for r, index in enumerate(selected):
                image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
                axes[r, 0].imshow(image)
                sx, sy = config["cosine_seed_xy"]
                axes[r, 0].scatter(
                    sx * config["image_size"],
                    sy * config["image_size"],
                    marker="+",
                    c="yellow",
                    s=60,
                )
                for j, d in enumerate(pair):
                    axes[r, 1 + j].imshow(d[desc["cosine"]][r], vmin=-1, vmax=1, cmap="coolwarm")
                    axes[r, 3 + j].imshow(d["rollout"][r], norm=roll_scale, cmap="viridis")
                for ax in axes[r]:
                    ax.set_xticks([])
                    ax.set_yticks([])
            for ax, title in zip(
                axes[0],
                [
                    "Input + seed",
                    "Cosine · no registers",
                    "Cosine · registers",
                    "Rollout · no registers",
                    "Rollout · registers",
                ],
            ):
                ax.set_title(title, fontsize=9)
            fig.colorbar(
                ScalarMappable(norm=Normalize(-1, 1), cmap="coolwarm"),
                ax=axes[:, 1:3].ravel().tolist(),
                shrink=0.25,
                label=f"Cosine · {space}",
            )
            fig.colorbar(
                ScalarMappable(norm=roll_scale, cmap="viridis"),
                ax=axes[:, 3:].ravel().tolist(),
                shrink=0.25,
                label="Rollout probability (log; feature-space independent)",
            )
            fig.suptitle(
                f"FeatLens · DINOv2 {size} · {desc['label']} cosine\nFull-token rollout is independent of displayed final feature space",
                fontsize=11,
            )
            name = f"featlens_comparison_{size}_{space}"
            save_figure(fig, results, config, name)
            if space == "normalized":
                _alias(results, name, f"featlens_comparison_{size}")
        prefix_distribution(config, results, size, pair[1])
    tail_diagnostic_figure(config, results, sources, image_dir)


def prefix_distribution(config: Config, results: Path, size: str, data: Any) -> None:
    """Place patch distributions beside CLS and each register across layers."""
    from .spaces import SPACES

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = ["black", "#4477aa", "#ee6677", "#228833", "#ccbb44"]
    for row, (space, desc) in enumerate(SPACES.items()):
        patches = data[desc["layers"]]
        prefix = data[desc["prefix"]]
        final = patches[:, -1].ravel()
        all_values = np.concatenate([final, prefix[:, -1].ravel()])
        bins = np.geomspace(max(all_values.min() * 0.9, 1e-6), all_values.max() * 1.1, 100)
        axes[row, 0].hist(
            final,
            bins=bins,
            histtype="step",
            label=f"Patches (n={final.size:,})",
            color="#888888",
            lw=2,
        )
        for token, color in enumerate(colors):
            name = "CLS" if token == 0 else f"Register {token}"
            axes[row, 0].hist(
                prefix[:, -1, token],
                bins=bins,
                histtype="step",
                label=f"{name} (n=20)",
                color=color,
                lw=1.3,
            )
        axes[row, 0].set(
            xscale="log",
            yscale="log",
            xlabel="Final token L2 norm (log)",
            ylabel="Token count (log)",
            title=desc["label"],
        )
        axes[row, 0].legend(fontsize=7)
        pooled = patches.transpose(1, 0, 2).reshape(patches.shape[1], -1)
        x = np.arange(len(pooled))
        lo, median, hi = np.quantile(pooled, [0.1, 0.5, 0.9], axis=1)
        axes[row, 1].fill_between(x, lo, hi, color="#bbbbbb", alpha=0.5, label="Patch 10–90%")
        axes[row, 1].plot(x, median, c="#777777", lw=2, label="Patch median")
        axes[row, 1].plot(
            x, np.quantile(pooled, 0.99, axis=1), c="#777777", ls=":", label="Patch 99th percentile"
        )
        for token, color in enumerate(colors):
            axes[row, 1].plot(
                x,
                prefix[:, :, token].mean(axis=0),
                c=color,
                label="CLS" if token == 0 else f"Register {token}",
            )
        axes[row, 1].set(
            yscale="log",
            xlabel="Block (0 = embedding)",
            ylabel=f"Token L2 norm · {space} (log)",
            title="Patch distribution and prefix-token means",
        )
        axes[row, 1].legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"DINOv2 {size} + registers · prefixes are measured separately and never reshaped as patches"
    )
    save_figure(fig, results, config, f"prefix_token_norms_{size}")


def tail_diagnostic_figure(
    config: Config, results: Path, sources: list[dict[str, Any]], image_dir: Path
) -> None:
    """Visualize shape, k sensitivity, and the image driving the pooled raw tail."""
    from .runtime import read_csv

    rows = read_csv(results / "data/large_tail_sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    colors = ["#cd593a", "#276987"]
    for suffix, color, label in zip(
        ["no_registers", "registers"], colors, ["Without registers", "With registers"]
    ):
        data = np.load(results / f"data/large_{suffix}.npz")
        raw = data["final_norms"]
        scaled = raw / np.median(raw)
        axes[0, 0].hist(
            scaled.ravel(),
            bins=np.linspace(0.5, 1.8, 100),
            histtype="step",
            color=color,
            label=label,
            lw=1.5,
        )
        r = [x for x in rows if x["model"] == f"large_{suffix}" and x["rule"] == "mean+k*std"]
        axes[0, 1].plot(
            [float(x["parameter"]) for x in r],
            [float(x["percent"]) for x in r],
            marker="o",
            color=color,
            label=label,
        )
    axes[0, 0].set(
        yscale="log",
        xlabel="Raw norm / own model median",
        ylabel="Patch count",
        title="Scale-adjusted raw distributions",
    )
    axes[0, 0].legend()
    axes[0, 1].set(
        xlabel="k in own pooled mean + k std",
        ylabel="Raw patch tail (%)",
        title="The rule changes the tail rate",
    )
    axes[0, 1].legend()
    data = np.load(results / "data/large_registers.npz")
    raw = data["final_norms"]
    mask = raw > raw.mean() + config["layer_k"] * raw.std()
    index = int(np.argmax(mask.sum(axis=1)))
    image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
    axes[1, 0].imshow(image)
    yy, xx = np.where(mask[index].reshape(37, 37))
    axes[1, 0].scatter(
        (xx + 0.5) * 14, (yy + 0.5) * 14, s=10, facecolors="none", edgecolors="#276987", lw=0.6
    )
    axes[1, 0].set_title(f"Raw pooled-tail patches · COCO {sources[index]['coco_id']:012d}")
    axes[1, 0].axis("off")
    for suffix, color, label in zip(
        ["no_registers", "registers"], colors, ["Without registers", "With registers"]
    ):
        d = np.load(results / f"data/large_{suffix}.npz")
        a = d["final_norms"]
        counts = (a > a.mean() + config["layer_k"] * a.std()).sum(axis=1)
        axes[1, 1].plot(np.arange(20), counts, marker="o", color=color, label=label)
    axes[1, 1].set(
        xlabel="Image index", ylabel="Raw pooled-tail patch count", title="Image concentration"
    )
    axes[1, 1].legend()
    fig.suptitle("Large raw-residual investigation · patch tokens only; CLS/registers excluded")
    save_figure(fig, results, config, "large_raw_tail_investigation")


def main_comparison(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> None:
    """Show the fixed Giant example in both spaces under separate row scales."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from .spaces import SPACES

    pair = [
        np.load(results / f"data/giant_{suffix}.npz") for suffix in ["no_registers", "registers"]
    ]
    index = config["hero_image_index"]
    image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), layout="constrained")
    for row, (space, desc) in enumerate(SPACES.items()):
        scale = Normalize(0, max(d[desc["norms"]][config["selected_indices"]].max() for d in pair))
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Input · {space}")
        for ax, d, label in zip(axes[row, 1:], pair, ["Without registers", "With 4 registers"]):
            norms = d[desc["norms"]][index].reshape(37, 37)
            ax.imshow(norms, cmap="magma", norm=scale, interpolation="nearest")
            ax.set_title(f"{label}\n{desc['label']}", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(
            ScalarMappable(norm=scale, cmap="magma"),
            ax=list(axes[row, 1:]),
            shrink=0.8,
            label=f"Patch L2 · {space}",
        )
    fig.suptitle("DINOv2 Giant · same color scale within each feature-space row")
    save_figure(fig, results, config, "main_comparison")

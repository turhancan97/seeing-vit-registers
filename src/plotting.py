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


def make_figures(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> None:
    """Render Experiments A–C and the FeatLens views with shared scales."""
    grid = config["image_size"] // config["patch_size"]
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib.cm import ScalarMappable

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    _, layer_rows, thresholds = analyze(config, sources, results)
    colors = ["#cd593a", "#276987"]
    for size in config["sizes"]:
        pair = [np.load(results / f"data/{size}_{s}.npz") for s in ["no_registers", "registers"]]
        threshold = thresholds[size]["threshold"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True, layout="constrained")
        mx = max(d["normalized_norms"].max() for d in pair)
        bins = np.linspace(0, mx * 1.03, config["histogram_bins"] + 1)
        for j, (ax, d, label) in enumerate(
            zip(axes, pair, ["Without registers", "With 4 registers"])
        ):
            ax.hist(d["normalized_norms"].ravel(), bins=bins, color=colors[j], alpha=0.9)
            ax.axvline(
                threshold,
                color="#303030",
                ls="--",
                lw=1.4,
                label=f"Shared threshold {threshold:.1f}",
            )
            ax.set(
                yscale="log",
                xlabel="Patch token L2 norm (HF last_hidden_state)",
                title=label,
                xlim=(0, max(mx * 1.03, threshold * 1.06)),
                ylim=(0.7, None),
            )
            n = d["normalized_norms"]
            ax.text(
                0.97,
                0.96,
                f"mean {n.mean():.1f}  |  max {n.max():.1f}\nabove cutoff: {(n > threshold).mean() * 100:.3f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )
            ax.legend(loc="center right", fontsize=8)
        axes[0].set_ylabel("Patch count (log scale)")
        fig.suptitle(
            f"DINOv2 {size} · {len(sources)} images · {len(sources) * grid**2:,} patches per model"
        )
        save_figure(fig, results, config, f"token_norm_histogram_{size}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
        for ax, space in zip(axes, ["final_norms", "normalized_norms"]):
            vals = [d[space].ravel() for d in pair]
            bins = np.linspace(0, max(v.max() for v in vals) * 1.03, config["histogram_bins"] + 1)
            for v, color, label in zip(vals, colors, ["Without registers", "With registers"]):
                ax.hist(v, bins=bins, histtype="step", lw=1.6, color=color, label=label)
            ax.set(
                yscale="log",
                xlabel="L2 norm",
                ylabel="Patch count",
                title="Before final LayerNorm"
                if space == "final_norms"
                else "HF last_hidden_state (after LayerNorm)",
            )
            ax.legend(fontsize=8)
        save_figure(fig, results, config, f"layernorm_diagnostic_{size}")

        fig, ax = plt.subplots(figsize=(8, 3.9), layout="constrained")
        for suffix, color, label in zip(
            ["no_registers", "registers"], colors, ["Without registers", "With registers"]
        ):
            rows = [r for r in layer_rows if r["model"] == f"{size}_{suffix}"]
            ax.plot(
                [r["layer"] for r in rows],
                [r["high_percent"] for r in rows],
                marker="o",
                ms=3,
                color=color,
                label=label,
            )
        ax.set(
            xlabel="Transformer block (0 = input embedding)",
            ylabel="Patch tokens above layer cutoff (%)",
            title=f"DINOv2 {size} · per-model, per-layer cutoff: mean + {config['layer_k']:g} × std",
        )
        ax.grid(alpha=0.18)
        ax.legend()
        save_figure(fig, results, config, f"high_norm_across_layers_{size}")

        selected = pair[0]["selected_indices"].tolist()
        rawmax = max(d["normalized_norms"][selected].max() for d in pair)
        attmax = max(d["cls_attention"].max() for d in pair)
        norm_scale = Normalize(0, rawmax)
        att_scale = LogNorm(config["attention_color_min"], attmax)
        fig, axes = plt.subplots(
            len(selected), 7, figsize=(17.5, 2.5 * len(selected)), layout="constrained"
        )
        titles = [
            "Input",
            "Norm · no registers",
            "Norm · registers",
            "CLS attention · no reg.",
            "CLS attention · registers",
            "PCA RGB · no registers",
            "PCA RGB · registers",
        ]
        for r, idx in enumerate(selected):
            im, _ = preprocess(image_dir / Path(sources[idx]["path"]).name, config)
            axes[r, 0].imshow(im)
            axes[r, 0].set_ylabel(Path(sources[idx]["path"]).stem, fontsize=8)
            for j, d in enumerate(pair):
                axes[r, 1 + j].imshow(
                    d["normalized_norms"][idx].reshape(grid, grid),
                    cmap="magma",
                    norm=norm_scale,
                    interpolation="nearest",
                )
                axes[r, 3 + j].imshow(
                    d["cls_attention"][r], cmap="viridis", norm=att_scale, interpolation="nearest"
                )
                axes[r, 5 + j].imshow(d["pca_rgb"][r], interpolation="nearest")
            for c, ax in enumerate(axes[r]):
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(titles[c], fontsize=10)
        fig.colorbar(
            ScalarMappable(norm=norm_scale, cmap="magma"),
            ax=axes[:, 1:3].ravel().tolist(),
            shrink=0.25,
            label="L2 norm after final LayerNorm",
        )
        fig.colorbar(
            ScalarMappable(norm=att_scale, cmap="viridis"),
            ax=axes[:, 3:5].ravel().tolist(),
            shrink=0.25,
            label="CLS → patch probability (log scale)",
        )
        fig.suptitle(
            f"DINOv2 {size} · shared norm (linear) and attention (log) scales across models and six images\nPCA fitted independently per image/model; colors have no shared semantic meaning",
            fontsize=12,
        )
        save_figure(fig, results, config, f"visual_comparison_{size}")

        # A compact blog-friendly image + two norm maps, with explicit high-norm markers.
        fig, axes = plt.subplots(
            len(selected), 3, figsize=(8.4, 2.7 * len(selected)), layout="constrained"
        )
        for r, idx in enumerate(selected):
            im, _ = preprocess(image_dir / Path(sources[idx]["path"]).name, config)
            axes[r, 0].imshow(im)
            for j, d in enumerate(pair):
                n = d["normalized_norms"][idx].reshape(grid, grid)
                axes[r, j + 1].imshow(n, cmap="magma", norm=norm_scale, interpolation="nearest")
                yy, xx = np.where(n > threshold)
                axes[r, 0].scatter(
                    (xx + 0.5) * config["patch_size"],
                    (yy + 0.5) * config["patch_size"],
                    s=12,
                    facecolors="none",
                    edgecolors=colors[j],
                    linewidths=0.7,
                )
                axes[r, j + 1].set_title(
                    f"{(n > threshold).sum()} / {grid**2} above {threshold:.1f}", fontsize=9
                )
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(
            axes[0], ["Input + high-norm patches", "Without registers", "With registers"]
        ):
            ax.set_title(title)
        fig.colorbar(
            ScalarMappable(norm=norm_scale, cmap="magma"),
            ax=axes[:, 1:].ravel().tolist(),
            shrink=0.3,
            label="Patch L2 norm",
        )
        fig.suptitle(
            f"DINOv2 {size} · high-norm locations (orange: no registers; blue: registers)",
            fontsize=11,
        )
        save_figure(fig, results, config, f"high_norm_locations_{size}")

        fig, axes = plt.subplots(
            len(selected), 5, figsize=(12.5, 2.5 * len(selected)), layout="constrained"
        )
        rollmax = max(d["rollout"].max() for d in pair)
        rollscale = LogNorm(config["attention_color_min"], rollmax)
        for r, idx in enumerate(selected):
            im, _ = preprocess(image_dir / Path(sources[idx]["path"]).name, config)
            axes[r, 0].imshow(im)
            sx, sy = config["cosine_seed_xy"]
            axes[r, 0].scatter(
                [sx * config["image_size"]],
                [sy * config["image_size"]],
                marker="+",
                color="yellow",
                s=70,
            )
            for j, d in enumerate(pair):
                axes[r, 1 + j].imshow(
                    d["cosine"][r], cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
                )
                axes[r, 3 + j].imshow(
                    d["rollout"][r], cmap="viridis", norm=rollscale, interpolation="nearest"
                )
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
        for ax, title in zip(
            axes[0],
            [
                "Input + cosine seed",
                "Cosine · no registers",
                "Cosine · registers",
                "Rollout · no registers",
                "Rollout · registers",
            ],
        ):
            ax.set_title(title, fontsize=10)
        fig.colorbar(
            ScalarMappable(norm=Normalize(-1, 1), cmap="coolwarm"),
            ax=axes[:, 1:3].ravel().tolist(),
            shrink=0.25,
            label="Cosine to center patch",
        )
        fig.colorbar(
            ScalarMappable(norm=rollscale, cmap="viridis"),
            ax=axes[:, 3:5].ravel().tolist(),
            shrink=0.25,
            label="CLS rollout probability (log scale)",
        )
        fig.suptitle(
            f"FeatLens · DINOv2 {size} · mean-head rollout, residual identity, no attention discarded",
            fontsize=11,
        )
        save_figure(fig, results, config, f"featlens_comparison_{size}")


def main_comparison(
    config: Config, sources: list[dict[str, Any]], results: Path, image_dir: Path
) -> None:
    """Make a compact README comparison using a CC-BY image from the six inputs."""
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    index = config["hero_image_index"]
    pair = [
        np.load(results / f"data/large_{suffix}.npz") for suffix in ["no_registers", "registers"]
    ]
    scale = Normalize(0, max(d["normalized_norms"][config["selected_indices"]].max() for d in pair))
    image, _ = preprocess(image_dir / Path(sources[index]["path"]).name, config)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    axes[0].imshow(image)
    axes[0].set_title(f"Input · COCO {sources[index]['coco_id']:012d}")
    grid = config["image_size"] // config["patch_size"]
    for ax, data, label in zip(axes[1:], pair, ["Without registers", "With 4 registers"]):
        norms = data["normalized_norms"][index].reshape(grid, grid)
        ax.imshow(norms, cmap="magma", norm=scale, interpolation="nearest")
        count = int((norms > config["threshold"]["value"]).sum())
        ax.set_title(f"{label}\n{count} high-norm patch" + ("es" if count != 1 else ""))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(
        ScalarMappable(norm=scale, cmap="magma"),
        ax=list(axes[1:]),
        shrink=0.8,
        label="Final patch L2 norm",
    )
    fig.suptitle("DINOv2 Large · 518 × 518 · same color scale")
    save_figure(fig, results, config, "main_comparison")

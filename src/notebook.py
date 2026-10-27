"""Small display helpers so notebooks contain no experiment implementation."""

from io import BytesIO
from pathlib import Path
import html
from PIL import Image
from IPython.display import HTML, Image as DisplayImage, Markdown, display
from .runtime import read_csv


def show_results(results: Path) -> None:
    """Display the final normalized-output table from the generated CSV."""
    rows = [
        r for r in read_csv(results / "data/summary.csv") if r["feature_space"] == "post_layernorm"
    ]
    columns = ["model", "mean_norm", "max_norm", "chosen_threshold", "high_count", "high_percent"]
    heading = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(r[c])}</td>" for c in columns) + "</tr>" for r in rows
    )
    display(HTML("<table>" + heading + body + "</table>"))


def show_figures(results: Path, names: list[str]) -> None:
    """Embed readable previews while preserving full-resolution PNG/PDF exports."""
    for name in names:
        with Image.open(results / f"figures/{name}.png") as source:
            preview = source.convert("RGB")
        preview.thumbnail((1400, 2200))
        buffer = BytesIO()
        preview.save(buffer, format="PNG")
        display(Markdown("### " + name.replace("_", " ")))
        display(DisplayImage(data=buffer.getvalue()))

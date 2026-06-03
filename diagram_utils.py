import base64
from pathlib import Path

import requests


def save_mermaid_assets(output_stem: Path, mermaid_text: str) -> None:
    """Save both Mermaid source and a rendered PNG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    mmd_path = output_stem.with_suffix(".mmd")
    png_path = output_stem.with_suffix(".png")

    mmd_path.write_text(mermaid_text, encoding="utf-8")

    encoded_graph = base64.urlsafe_b64encode(mermaid_text.encode("utf-8")).decode("ascii")
    image_url = f"https://mermaid.ink/img/{encoded_graph}"
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    png_path.write_bytes(response.content)

#!/usr/bin/env python3
"""
Render Mermaid diagrams via Kroki and save as PNG files for LaTeX.
"""

from __future__ import annotations

from pathlib import Path
from urllib import request


KROKI_ENDPOINT = "https://kroki.io/mermaid/png"


def render_mermaid(input_path: Path, output_path: Path) -> None:
    payload = input_path.read_text(encoding="utf-8").encode("utf-8")
    req = request.Request(
        KROKI_ENDPOINT,
        data=payload,
        headers={
            "Content-Type": "text/plain",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "image/png",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    print(f"Wrote {output_path} ({len(data)} bytes)")


def main() -> None:
    base = Path(__file__).resolve().parent
    mappings = {
        base / "figures/system_architecture.mmd": base / "plots/system_architecture.png",
        base / "figures/research_pipeline.mmd": base / "plots/trial_state_machine.png",
    }
    for src, dst in mappings.items():
        render_mermaid(src, dst)


if __name__ == "__main__":
    main()

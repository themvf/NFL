from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from nfl_app.ingest.fetch import IngestConfig, fetch_and_store

app = typer.Typer(add_completion=False)


@app.command()
def run(
    season: Optional[List[int]] = typer.Option(
        None, "--season", "-s", help="Repeat for each season, e.g. -s 2023 -s 2024"
    ),
    data_dir: Path = typer.Option(Path("data/processed"), "--data-dir", help="Output directory"),
):
    cfg = IngestConfig(seasons=season or [], data_dir=data_dir)
    res = fetch_and_store(cfg)
    print(res)


if __name__ == "__main__":
    app()

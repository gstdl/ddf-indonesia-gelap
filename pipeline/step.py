"""Run one box of the pipeline.

A box is a module with `run(*tables) -> table`. Its output is saved to data/<box path>.parquet,
e.g. pipeline/layer1_prepare/normalize.py -> data/layer1_prepare/normalize.parquet.
Writes are atomic: a failed box never leaves a partial file. Each box run is logged with its
inputs, output size, duration and file.

Idempotent: a box is skipped when its output exists and is newer than every input (input boxes' outputs
and input files passed as Path params). A box rerun therefore makes every box downstream of it rerun.
Boxes named in OVERWRITE (module labels such as "layer1_prepare/image") always rerun.
"""

import logging
import os
import time
from pathlib import Path
from types import ModuleType

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

log = logging.getLogger(__name__)

# Module labels of boxes to rerun even when their output is up to date; set by main.py --overwrite.
OVERWRITE: set[str] = set()


def box_name(box: ModuleType, name: str | None = None) -> str:
    parts = box.__name__.split(".")[1:]
    return "/".join(parts[:-1] + [name or parts[-1]])


def output_path(box: ModuleType, name: str | None = None) -> Path:
    parts = box.__name__.split(".")[1:]
    if name is not None:
        parts[-1] = name
    return DATA.joinpath(*parts).with_suffix(".parquet")


# Long loops log a progress line every this many items.
PROGRESS_EVERY = 25


def progress(items: list, label: str):
    """Yield `items`, logging every PROGRESS_EVERY items how far the loop is and the estimated time left."""
    started = time.perf_counter()
    for done, item in enumerate(items, start=1):
        yield item
        if done % PROGRESS_EVERY == 0 or done == len(items):
            per_item = (time.perf_counter() - started) / done
            log.info("%s: %d/%d (%.1fs each, ~%.0f min left)", label, done, len(items), per_item,
                     per_item * (len(items) - done) / 60)


def step(box: ModuleType, *inputs: ModuleType, name: str | None = None, **params) -> None:
    """Run `box` on the saved outputs of `inputs`, plus keyword `params` such as an input file path.
    `name` renames the output file when a box is used twice."""
    label = box_name(box, name)
    sources = [box_name(source) for source in inputs] + [f"{key}={value}" for key, value in params.items()]
    path = output_path(box, name)
    # Boxes given params (e.g. the input CSV path) record them; a different value makes the box rerun.
    params_path = path.with_name(path.name + ".params")
    params_text = repr(sorted((key, str(value)) for key, value in params.items()))
    params_match = params_path.exists() and params_path.read_text() == params_text if params else True
    input_files = [output_path(source) for source in inputs] + [v for v in params.values() if isinstance(v, Path)]
    if (
        box_name(box) not in OVERWRITE
        and path.exists()
        and params_match
        and all(path.stat().st_mtime > file.stat().st_mtime for file in input_files)
    ):
        log.info("skip  %s: up to date -> %s", label, path.relative_to(DATA.parent))
        return
    log.info("start %s <- %s", label, ", ".join(sources) or "(no inputs)")
    started = time.perf_counter()
    table = box.run(*[pd.read_parquet(output_path(source)) for source in inputs], **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    table.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)
    if params:
        params_path.write_text(params_text)
    log.info(
        "done  %s: %d rows x %d columns in %.1fs -> %s",
        label, len(table), table.shape[1], time.perf_counter() - started, path.relative_to(DATA.parent),
    )

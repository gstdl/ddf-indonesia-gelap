import types

import pandas as pd

from pipeline import step as step_module


def make_box(name: str, calls: list) -> types.ModuleType:
    box = types.ModuleType(f"pipeline.testlayer.{name}")

    def run(*tables, **params):
        calls.append(name)
        return pd.DataFrame({"n": [len(tables)]})

    box.run = run
    return box


def test_skips_up_to_date_boxes_and_reruns_overwritten_ones_downstream(tmp_path, monkeypatch):
    monkeypatch.setattr(step_module, "DATA", tmp_path)
    monkeypatch.setattr(step_module, "OVERWRITE", set())
    calls = []
    source = tmp_path / "input.csv"
    source.write_text("x")
    first, second = make_box("first", calls), make_box("second", calls)

    step_module.step(first, path=source)
    step_module.step(second, first)
    assert calls == ["first", "second"]

    step_module.step(first, path=source)
    step_module.step(second, first)
    assert calls == ["first", "second"]  # both up to date: skipped

    step_module.OVERWRITE.add("testlayer/first")
    step_module.step(first, path=source)
    step_module.step(second, first)
    assert calls == ["first", "second", "first", "second"]  # overwrite reruns first; second's input is newer


def test_different_input_file_reruns_the_box(tmp_path, monkeypatch):
    monkeypatch.setattr(step_module, "DATA", tmp_path)
    monkeypatch.setattr(step_module, "OVERWRITE", set())
    calls = []
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    new.write_text("x")
    old.write_text("x")
    box = make_box("first", calls)

    step_module.step(box, path=new)
    step_module.step(box, path=old)
    assert calls == ["first", "first"]

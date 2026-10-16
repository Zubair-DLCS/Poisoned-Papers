"""End-to-end check with a deliberately gullible mock model (no API key, no network)."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from run import run
from analyze import analyse
from data import sample, criteria
from agent import parse, messages
from attacks import inject, VARIED
from guard import flag


def test_parse():
    assert parse('{"decision": "include", "reason": "x"}') == "include"
    assert parse('Sure. {"decision":"EXCLUDE"}') == "exclude"
    assert parse("I would exclude it.") == "exclude"
    assert parse("no idea") == "invalid"


def test_spotlight_marks_data():
    import random
    m = messages({"review": "r", "criteria": "c", "topic": "t"}, "A title", "Some text here",
                 "spotlight", random.Random(1))
    assert "^" in m[1]["content"] and m[1]["content"].startswith("<record_")


def test_guard_split():
    assert flag(inject("An abstract.", "authority", "t"))
    assert not flag("We propose a defect prediction model evaluated on NASA datasets.")
    assert len(VARIED) == 10


def test_end_to_end_with_mock():
    with tempfile.TemporaryDirectory() as tmp:
        d = run("mock", "gullible", per_class=2, limit=None, res_dir=tmp)
        assert set(d.defence) == {"none", "warning", "spotlight"}
        perf, asr = analyse(d, sample(2), criteria())
        assert len(perf) == 3 and len(asr) > 0
        naive = asr[(asr.attack == "naive") & (asr.defence == "none")]
        assert naive.success.iloc[0] == 1.0          # the mock obeys plain commands
        warn = asr[(asr.attack == "naive") & (asr.defence == "warning")]
        assert warn.success.iloc[0] == 0.0           # ...unless warned


if __name__ == "__main__":
    for f in (test_parse, test_spotlight_marks_data, test_guard_split, test_end_to_end_with_mock):
        f(); print("ok", f.__name__)

"""Agent-style tool templates (src/agent_tools.py): every tool renders, fills its intents and clean-A templates,
has a middle insertion boundary, and the clean-A partition puts at least one template of each tool in each split."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agent_tools import GENERATORS, INTENTS, TOOLS, clean_a_templates, fill  # noqa: E402
from probes import V3_BOUNDARIES, has_internal_boundary  # noqa: E402


def test_at_least_15_tools():
    assert len(TOOLS) >= 15


def test_every_tool_renders_and_fills():
    r = random.Random(0)
    templates = clean_a_templates()
    for tool, gen in GENERATORS.items():
        for _ in range(20):
            o = gen(r)
            assert {"name", "day", "item"} <= set(o.fields), tool
            assert has_internal_boundary(o.text, V3_BOUNDARIES), tool
            for it in INTENTS[tool]:
                fill(it, o.fields)
            for _, t, s in templates:
                if t in (tool, None):
                    fill(s, o.fields)


def test_generation_is_deterministic():
    a = [GENERATORS[t](random.Random(f"x:{t}")).text for t in TOOLS]
    b = [GENERATORS[t](random.Random(f"x:{t}")).text for t in TOOLS]
    assert a == b


def test_clean_a_partition_covers_both_splits():
    split = lambda tid: "val" if int(tid.rsplit(":", 1)[1]) % 4 == 3 else "train"
    for tool in TOOLS:
        ids = [tid for tid, t, _ in clean_a_templates() if t == tool]
        assert {split(i) for i in ids} == {"train", "val"}, tool

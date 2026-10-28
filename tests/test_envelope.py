import json

from airport_agent.agent.envelope import Fact, ToolResult


def test_tool_result_round_trip():
    r = ToolResult(
        data={"x": 1},
        facts={"SFO.load_factor": Fact(value=0.85, unit="pct", source="t100")},
    )
    payload = json.loads(r.to_llm_json())
    assert payload["facts"]["SFO.load_factor"] == {"value": 0.85, "unit": "pct", "source": "t100"}
    assert payload["error"] is None


def test_numpy_values_serialize():
    import numpy as np

    r = ToolResult(data={"n": np.int64(3), "x": np.float64(0.5)})
    assert r.to_plain()["data"] == {"n": 3, "x": 0.5}


def test_defaults_are_not_shared():
    a, b = ToolResult(), ToolResult()
    a.assumptions.append("x")
    assert b.assumptions == []

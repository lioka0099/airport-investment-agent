"""The common shape every tool returns. Facts are the only numbers the answer may cite."""

import json

from pydantic import BaseModel, Field


def _default(o):
    # pandas hands back numpy scalars; .item() turns them into plain Python numbers
    return o.item() if hasattr(o, "item") else str(o)


class Fact(BaseModel):
    value: float | str
    unit: str  # pct, pctl, count, minutes, score, rank, ratio, miles, hhi, text
    source: str  # key of config.SOURCES


class ToolResult(BaseModel):
    data: dict = Field(default_factory=dict)
    facts: dict[str, Fact] = Field(default_factory=dict)
    findings: list[dict] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_coverage: dict = Field(default_factory=dict)
    error: dict | None = None

    def to_llm_json(self) -> str:
        return json.dumps(self.model_dump(), default=_default, separators=(",", ":"))

    def to_plain(self) -> dict:
        return json.loads(self.to_llm_json())

from airport_agent import config


def test_weights_sum_to_one():
    assert abs(sum(config.WEIGHTS.values()) - 1.0) < 1e-9


def test_weights_cover_all_components():
    assert set(config.WEIGHTS) == {"demand", "growth", "terminal", "airside", "scale"}


def test_new_england_region():
    assert config.REGIONS["new england"] == ["CT", "ME", "MA", "NH", "RI", "VT"]


def test_la_metro_primary_and_alternatives():
    la = config.METROS["la"]
    assert la["primary"] == ["LAX"]
    assert "SNA" in la["alternatives"]


def test_reviewer_chain_has_three_providers():
    assert [p for p, _ in config.REVIEWER_CHAIN] == ["anthropic", "google", "openai"]

import httpx
import pytest

from airport_agent.data.http import get_with_retry


def _client(statuses):
    calls = iter(statuses)

    def handler(request):
        return httpx.Response(next(calls), text="ok")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retries_5xx_then_succeeds():
    sleeps = []
    resp = get_with_retry("https://x", client=_client([503, 502, 200]), sleep=sleeps.append)
    assert resp.status_code == 200
    assert sleeps == [1, 2]


def test_4xx_raises_without_retry():
    sleeps = []
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry("https://x", client=_client([404]), sleep=sleeps.append)
    assert sleeps == []


def test_gives_up_after_retries():
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry(
            "https://x", retries=2, client=_client([500, 500, 500]), sleep=lambda s: None
        )

"""GET with timeout and exponential-backoff retry, for the offline ingest job."""

import time

import httpx


def get_with_retry(url: str, params: dict | None = None, *, timeout: float = 60, retries: int = 3,
                   client: httpx.Client | None = None, sleep=time.sleep) -> httpx.Response:
    client = client or httpx.Client(follow_redirects=True)
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, params=params, timeout=timeout)
            if resp.status_code < 500:
                resp.raise_for_status()  # 4xx is our bug or a missing file: no retry
                return resp
            if attempt == retries:
                resp.raise_for_status()
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt == retries:
                raise
        sleep(2**attempt)
    raise AssertionError("unreachable")

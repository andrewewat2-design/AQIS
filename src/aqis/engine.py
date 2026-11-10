from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any


class EngineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EngineStatus:
    healthy: bool
    state: str
    detail: str


class TradingEngineAdapter:
    """Narrow adapter contract; AI agents never receive the raw engine API."""

    def health(self) -> EngineStatus:
        raise NotImplementedError

    def pause(self) -> None:
        raise NotImplementedError

    def cancel_pending_orders(self) -> None:
        raise NotImplementedError


class FreqtradeAdapter(TradingEngineAdapter):
    """Restricted client for an AQIS-managed Freqtrade dry-run instance.

    This client deliberately has no force-entry, force-exit, config-reload, or
    live-mode methods. The AQIS risk gate must still be integrated at the
    Freqtrade strategy's final order-confirmation callbacks before use.
    """

    def __init__(self, base_url: str, username: str, password: str, timeout: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._token: str | None = None

    def health(self) -> EngineStatus:
        try:
            response = self._request("GET", "/api/v1/health", authenticated=True)
            return EngineStatus(True, "running", json.dumps(response, sort_keys=True))
        except (EngineUnavailable, HTTPError, URLError, TimeoutError) as error:
            return EngineStatus(False, "unavailable", str(error))

    def pause(self) -> None:
        self._request("POST", "/api/v1/pause", authenticated=True)

    def cancel_pending_orders(self) -> None:
        trades = self._request("GET", "/api/v1/status", authenticated=True)
        for trade in trades if isinstance(trades, list) else []:
            trade_id = trade.get("trade_id")
            if trade_id is not None and trade.get("has_open_orders"):
                self._request("DELETE", f"/api/v1/trades/{trade_id}/open-order", authenticated=True)

    def _request(self, method: str, path: str, authenticated: bool = False) -> Any:
        headers = {"Accept": "application/json"}
        if authenticated:
            self._ensure_token()
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(f"{self.base_url}{path}", headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured local adapter endpoint
                body = response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            raise EngineUnavailable(f"Freqtrade request {method} {path} failed: {error}") from error
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _ensure_token(self) -> None:
        if self._token:
            return
        credentials = f"{self.username}:{self.password}".encode("utf-8")
        import base64

        request = Request(
            f"{self.base_url}/api/v1/token/login",
            data=b"",
            headers={
                "Authorization": f"Basic {base64.b64encode(credentials).decode('ascii')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured local adapter endpoint
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise EngineUnavailable(f"Freqtrade authentication failed: {error}") from error
        self._token = str(payload["access_token"])

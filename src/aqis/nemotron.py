"""Nemotron Research Agent — shadow-mode advisory component.

Provides research and analysis capabilities without influencing the
selector, risk engine, or execution system. All output is advisory and
cannot place orders, modify positions, change risk limits, or resume a
pause.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FindingKind(str, Enum):
    MARKET_FACT = "market_fact"
    MODEL_OPINION = "model_opinion"
    RECOMMENDATION = "recommendation"


@dataclass(frozen=True, slots=True)
class Finding:
    """A single research finding, classified as fact, opinion, or recommendation.

    A market_fact must carry an evidence_ref pointing into the supplied input
    or a trusted source; unverified assertions are reclassified as model_opinion.
    """

    kind: FindingKind
    content: str
    source: str
    confidence: float | None = None
    evidence_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """Structured output from the Nemotron Research Agent."""

    analysis_id: str
    model_name: str
    model_version: str
    timestamp_utc: datetime
    topic: str
    findings: tuple[Finding, ...]
    summary: str
    confidence: float
    raw_response: dict[str, Any]


class NemotronProvider(Protocol):
    """Provider abstraction so the underlying model can be changed later."""

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, str]: ...


class OpenAICompatibleNemotronProvider:
    """NVIDIA NIM-compatible chat provider for Nemotron models.

    Reads configuration from environment variables only. Credentials are
    never logged or included in audit reports.
    """

    def __init__(self, endpoint: str, api_key: str, model: str, timeout: int = 30) -> None:
        self.endpoint = endpoint
        self._api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> OpenAICompatibleNemotronProvider | None:
        endpoint = os.getenv("NEMOTRON_API_URL")
        api_key = os.getenv("NEMOTRON_API_KEY")
        model = os.getenv("NEMOTRON_MODEL")
        if not all((endpoint, api_key, model)):
            return None
        return cls(endpoint, api_key, model)

    def model_info(self) -> dict[str, str]:
        return {
            "model_name": self.model,
            "model_version": "nvidia-nim",
            "base_url": self.endpoint,
        }

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleNemotronProvider(endpoint={self.endpoint!r}, "
            f"model={self.model!r}, timeout={self.timeout})"
        )

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - configured provider endpoint
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("agent response must be a JSON object")
        return parsed


class MockNemotronProvider:
    """Deterministic offline provider for testing and fallback operation.

    Returns schema-valid canned responses. Configurable to simulate API
    failures and malformed responses.
    """

    def __init__(self, fail: bool = False, malformed: bool = False) -> None:
        self.fail = fail
        self.malformed = malformed
        self.call_count = 0
        self.last_user: str | None = None

    def model_info(self) -> dict[str, str]:
        return {
            "model_name": "mock-nemotron",
            "model_version": "0.1.0",
            "base_url": "mock://local",
        }

    def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        self.call_count += 1
        self.last_user = user
        if self.fail:
            raise URLError("mock provider failure")
        if self.malformed:
            return {"confidence": 1.5, "findings": [], "summary": "malformed"}
        topic = self._infer_topic(user)
        return {
            "findings": [
                {
                    "kind": "market_fact",
                    "content": f"Mock market observation for {topic}.",
                    "source": "mock_provider",
                    "evidence_ref": "input",
                },
                {
                    "kind": "model_opinion",
                    "content": f"Model suggests caution in {topic} conditions.",
                    "source": "mock_provider",
                    "confidence": 0.6,
                },
                {
                    "kind": "recommendation",
                    "content": f"Recommend further validation before acting on {topic}.",
                    "source": "mock_provider",
                    "confidence": 0.5,
                },
            ],
            "summary": f"Mock research summary for {topic}.",
            "confidence": 0.55,
        }

    @staticmethod
    def _infer_topic(user: str) -> str:
        lower = user.lower()
        for key in ("market", "portfolio", "disagreement", "unusual", "rejected", "decision", "experiment"):
            if key in lower:
                return key
        return "general"


class NemotronResearchAgent:
    """Shadow-mode research agent. Advisory only; no execution tools."""

    def __init__(self, provider: NemotronProvider | None = None) -> None:
        if provider is None:
            provider = OpenAICompatibleNemotronProvider.from_environment() or MockNemotronProvider()
        self.provider = provider

    # ------------------------------------------------------------------
    # Public research entry points (all shadow mode, advisory only)
    # ------------------------------------------------------------------

    def research_market_summary(self, market_data: dict[str, Any]) -> ResearchReport:
        return self._research(
            topic="market_summary",
            system=self._system_prompt("market_summary"),
            user=json.dumps(market_data, sort_keys=True),
        )

    def research_portfolio_performance(self, portfolio_snapshot: dict[str, Any]) -> ResearchReport:
        return self._research(
            topic="portfolio_performance",
            system=self._system_prompt("portfolio_performance"),
            user=json.dumps(portfolio_snapshot, sort_keys=True),
        )

    def investigate_strategy_disagreement(
        self, signals: list[dict[str, Any]], selection: dict[str, Any]
    ) -> ResearchReport:
        return self._research(
            topic="strategy_disagreement",
            system=self._system_prompt("strategy_disagreement"),
            user=json.dumps({"signals": signals, "selection": selection}, sort_keys=True),
        )

    def identify_unusual_conditions(
        self, regime: dict[str, Any], monitoring: dict[str, Any]
    ) -> ResearchReport:
        return self._research(
            topic="unusual_conditions",
            system=self._system_prompt("unusual_conditions"),
            user=json.dumps({"regime": regime, "monitoring": monitoring}, sort_keys=True),
        )

    def review_rejected_orders(
        self, rejections: list[dict[str, Any]], interventions: list[dict[str, Any]]
    ) -> ResearchReport:
        return self._research(
            topic="rejected_orders",
            system=self._system_prompt("rejected_orders"),
            user=json.dumps({"rejections": rejections, "interventions": interventions}, sort_keys=True),
        )

    def explain_trading_decision(
        self, correlation_id: str, events: list[dict[str, Any]]
    ) -> ResearchReport:
        return self._research(
            topic="explain_decision",
            system=self._system_prompt("explain_decision"),
            user=json.dumps({"correlation_id": correlation_id, "events": events}, sort_keys=True),
        )

    def recommend_strategy_experiments(
        self, strategy_performance: dict[str, Any]
    ) -> ResearchReport:
        return self._research(
            topic="strategy_experiments",
            system=self._system_prompt("strategy_experiments"),
            user=json.dumps(strategy_performance, sort_keys=True),
        )

    # ------------------------------------------------------------------
    # Internal: call provider, validate, build report
    # ------------------------------------------------------------------

    def _research(self, topic: str, system: str, user: str) -> ResearchReport:
        model_info = self.provider.model_info()
        timestamp = datetime.now(timezone.utc)
        input_fingerprint = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
        analysis_id = self._analysis_id(topic, model_info, input_fingerprint)

        try:
            raw_response = self.provider.complete_json(system=system, user=user)
            self._validate_response(raw_response)
        except (URLError, HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
            return self._safe_fallback(
                topic=topic,
                model_info=model_info,
                timestamp=timestamp,
                analysis_id=analysis_id,
                reason=str(error),
            )

        return self._build_report(
            topic=topic,
            model_info=model_info,
            timestamp=timestamp,
            analysis_id=analysis_id,
            raw_response=raw_response,
        )

    @staticmethod
    def _analysis_id(topic: str, model_info: dict[str, str], input_fingerprint: str) -> str:
        material = f"{topic}:{model_info.get('model_name','')}:{model_info.get('model_version','')}:{input_fingerprint}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _validate_response(response: dict[str, Any]) -> None:
        if not isinstance(response, dict):
            raise ValueError("agent response must be a JSON object")
        findings = response.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError("findings must be a list")
        summary = response.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        confidence = response.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be a number in [0, 1]")
        # Semantic validation: an empty response (no findings and no summary)
        # provides no research value and must be treated as invalid.
        if not findings and not summary:
            raise ValueError("agent response must include at least one finding or a non-empty summary")
        for finding in findings:
            if not isinstance(finding, dict):
                raise ValueError("each finding must be a JSON object")
            if "kind" not in finding or "content" not in finding:
                raise ValueError("each finding must have kind and content")
            if finding["kind"] not in {kind.value for kind in FindingKind}:
                raise ValueError(f"invalid finding kind: {finding['kind']}")
            fc = finding.get("confidence")
            if fc is not None and not 0 <= float(fc) <= 1:
                raise ValueError("finding confidence must be in [0, 1]")
            er = finding.get("evidence_ref")
            if er is not None and not isinstance(er, str):
                raise ValueError("finding evidence_ref must be a string")

    @classmethod
    def _build_report(
        cls,
        topic: str,
        model_info: dict[str, str],
        timestamp: datetime,
        analysis_id: str,
        raw_response: dict[str, Any],
    ) -> ResearchReport:
        findings: list[Finding] = []
        for item in raw_response.get("findings", []):
            fc = item.get("confidence")
            er = item.get("evidence_ref")
            kind = FindingKind(item["kind"])
            # A market_fact must be traceable to supplied evidence; otherwise it is
            # a model-generated opinion, not a verified fact.
            if kind is FindingKind.MARKET_FACT and not er:
                kind = FindingKind.MODEL_OPINION
            findings.append(
                Finding(
                    kind=kind,
                    content=str(item["content"]),
                    source=str(item.get("source", "model")),
                    confidence=float(fc) if fc is not None else None,
                    evidence_ref=str(er) if er is not None else None,
                )
            )
        confidence = raw_response.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            confidence = 0.0
        return ResearchReport(
            analysis_id=analysis_id,
            model_name=model_info.get("model_name", "unknown"),
            model_version=model_info.get("model_version", "unknown"),
            timestamp_utc=timestamp,
            topic=topic,
            findings=tuple(findings),
            summary=str(raw_response.get("summary", "")),
            confidence=float(confidence),
            raw_response=dict(raw_response),
        )

    @classmethod
    def _safe_fallback(
        cls,
        topic: str,
        model_info: dict[str, str],
        timestamp: datetime,
        analysis_id: str,
        reason: str,
    ) -> ResearchReport:
        # Never propagate raw exception text into audit-visible output: it may
        # contain credentials leaked by the provider. Use a safe generic reason.
        return ResearchReport(
            analysis_id=analysis_id,
            model_name=model_info.get("model_name", "unknown"),
            model_version=model_info.get("model_version", "unknown"),
            timestamp_utc=timestamp,
            topic=topic,
            findings=(
                Finding(
                    kind=FindingKind.MODEL_OPINION,
                    content=(
                        "Nemotron provider unavailable; report generated from deterministic fallback. "
                        f"Reason: {reason}"
                    ),
                    source="fallback",
                    confidence=0.0,
                ),
            ),
            summary=f"Provider unavailable for {topic}; no model opinion formed.",
            confidence=0.0,
            raw_response={"error": "provider_unavailable", "fallback": True},
        )

    # ------------------------------------------------------------------
    # System prompts (advisory, shadow mode, no execution authority)
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt(topic: str) -> str:
        base = (
            "You are the AQIS Nemotron Research Agent. You operate in shadow mode only: observe, "
            "analyse, and recommend. You cannot place orders, modify positions, change risk limits, "
            "approve your own recommendations, or resume a pause. Return a single JSON object with "
            "keys: findings (array of {kind, content, source, confidence?}), summary (string), "
            "confidence (0..1). Classify each finding kind as market_fact (verified market data), "
            "model_opinion (your analysis), or recommendation (advisory only). Clearly separate "
            "model-generated opinions from verified market facts. Do not include credentials, keys, "
            "or secrets in any field."
        )
        topic_prompts = {
            "market_summary": base + " Analyse the supplied market summary (prices, volatility, volume, regime).",
            "portfolio_performance": base + " Examine the supplied portfolio performance snapshot (equity, exposure, drawdown, fees, slippage).",
            "strategy_disagreement": base + " Investigate disagreement between strategy agents' signals and the selector outcome.",
            "unusual_conditions": base + " Identify unusual market conditions from regime and monitoring data.",
            "rejected_orders": base + " Review rejected orders and risk interventions; explain why each was rejected.",
            "explain_decision": base + " Explain a trading decision given its correlation ID and linked audit events.",
            "strategy_experiments": base + " Recommend strategy experiments based on performance data. Every experiment must pass AQIS backtest, validation, and human approval before use.",
        }
        return topic_prompts.get(topic, base)

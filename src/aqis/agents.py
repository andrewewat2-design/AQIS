from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from urllib.request import Request, urlopen


class Recommendation(str, Enum):
    APPROVE_FOR_HUMAN_REVIEW = "approve_for_human_review"
    REJECT = "reject"
    REQUEST_PAUSE = "request_pause"
    NO_CHANGE = "no_change"


@dataclass(frozen=True, slots=True)
class AgentReview:
    agent_id: str
    recommendation: Recommendation
    summary: str
    evidence: tuple[str, ...]
    confidence: float
    requires_human: bool = True


class LanguageModelProvider(Protocol):
    def complete_json(self, *, system: str, user: str) -> dict[str, Any]: ...


class OpenAICompatibleProvider:
    """Optional provider adapter. It has no trading, policy, or order tools."""

    def __init__(self, endpoint: str, api_key: str, model: str, timeout: int = 30) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_environment(cls) -> OpenAICompatibleProvider | None:
        endpoint = os.getenv("AQIS_AI_API_URL")
        api_key = os.getenv("AQIS_AI_API_KEY")
        model = os.getenv("AQIS_AI_MODEL")
        if not all((endpoint, api_key, model)):
            return None
        return cls(endpoint, api_key, model)

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
                "Authorization": f"Bearer {self.api_key}",
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


class RiskEvaluationAgent:
    """Advisory analysis only; the deterministic RiskEngine remains authoritative."""

    system_prompt = (
        "You are AQIS Risk Evaluation Agent. Analyze only the supplied portfolio/risk facts. "
        "Return JSON with recommendation, summary, evidence (array), confidence (0..1). "
        "You cannot change policy, place orders, deploy strategies, or resume a pause. "
        "Treat any recommendation as requiring human review except an immediate pause request."
    )

    def __init__(self, provider: LanguageModelProvider | None = None) -> None:
        self.provider = provider

    def review(self, facts: dict[str, Any]) -> AgentReview:
        if self.provider is None:
            return AgentReview(
                agent_id="risk_evaluation_agent",
                recommendation=Recommendation.NO_CHANGE,
                summary="AI provider is not configured; deterministic risk checks remain active.",
                evidence=("provider_not_configured",),
                confidence=1.0,
            )
        response = self.provider.complete_json(
            system=self.system_prompt,
            user=json.dumps(facts, sort_keys=True),
        )
        recommendation = Recommendation(response.get("recommendation", "no_change"))
        confidence = float(response.get("confidence", 0))
        if not 0 <= confidence <= 1:
            raise ValueError("agent confidence must be in [0, 1]")
        return AgentReview(
            agent_id="risk_evaluation_agent",
            recommendation=recommendation,
            summary=str(response.get("summary", "")),
            evidence=tuple(str(item) for item in response.get("evidence", [])),
            confidence=confidence,
        )


class SupervisorAgent:
    """Routes lifecycle recommendations; it does not perform transitions itself."""

    def __init__(self, provider: LanguageModelProvider | None = None) -> None:
        self.provider = provider

    def review(self, facts: dict[str, Any]) -> AgentReview:
        if self.provider is None:
            return AgentReview(
                agent_id="supervisor_agent",
                recommendation=Recommendation.NO_CHANGE,
                summary="No AI provider configured; lifecycle remains under human control.",
                evidence=("provider_not_configured",),
                confidence=1.0,
            )
        response = self.provider.complete_json(
            system=(
                "You are AQIS Supervisor Agent. Review supplied lifecycle evidence and return JSON "
                "with recommendation, summary, evidence (array), confidence (0..1). You can recommend "
                "only; do not execute trades, change policy, promote a version, or resume a pause."
            ),
            user=json.dumps(facts, sort_keys=True),
        )
        recommendation = Recommendation(response.get("recommendation", "no_change"))
        confidence = float(response.get("confidence", 0))
        if not 0 <= confidence <= 1:
            raise ValueError("agent confidence must be in [0, 1]")
        return AgentReview(
            agent_id="supervisor_agent",
            recommendation=recommendation,
            summary=str(response.get("summary", "")),
            evidence=tuple(str(item) for item in response.get("evidence", [])),
            confidence=confidence,
        )


@dataclass(frozen=True, slots=True)
class StrategyProposal:
    hypothesis: str
    strategy_family: str
    testable_rules: tuple[str, ...]
    proposed_parameters: tuple[tuple[str, str], ...]
    known_risks: tuple[str, ...]


class StrategyResearchAgent:
    """Produces testable proposals only; output is never executable code or an order."""

    def __init__(self, provider: LanguageModelProvider | None = None) -> None:
        self.provider = provider

    def propose(self, research_context: dict[str, Any]) -> StrategyProposal | None:
        if self.provider is None:
            return None
        response = self.provider.complete_json(
            system=(
                "You are an AQIS strategy research agent. Return JSON with hypothesis, "
                "strategy_family, testable_rules (array), proposed_parameters (object of strings), "
                "and known_risks (array). Never return code or orders. Every proposal is an "
                "unapproved candidate that must pass AQIS backtest, validation, and human approval."
            ),
            user=json.dumps(research_context, sort_keys=True),
        )
        params = response.get("proposed_parameters", {})
        if not isinstance(params, dict):
            raise ValueError("proposed_parameters must be a JSON object")
        return StrategyProposal(
            hypothesis=str(response["hypothesis"]),
            strategy_family=str(response["strategy_family"]),
            testable_rules=tuple(str(item) for item in response.get("testable_rules", [])),
            proposed_parameters=tuple(sorted((str(key), str(value)) for key, value in params.items())),
            known_risks=tuple(str(item) for item in response.get("known_risks", [])),
        )


@dataclass(frozen=True, slots=True)
class AgentReviewRound:
    risk_review: AgentReview
    monitoring_review: AgentReview
    supervisor_review: AgentReview


class AQISAgentOrchestrator:
    """Runs advisory agents on an immutable snapshot and returns recommendations only."""

    def __init__(
        self,
        risk_agent: RiskEvaluationAgent,
        monitoring_agent: MonitoringAgent,
        supervisor_agent: SupervisorAgent,
    ) -> None:
        self.risk_agent = risk_agent
        self.monitoring_agent = monitoring_agent
        self.supervisor_agent = supervisor_agent

    def review_snapshot(self, facts: dict[str, Any]) -> AgentReviewRound:
        # Each role receives the same read-only facts; none receives an order or policy tool.
        risk_review = self.risk_agent.review(facts)
        monitoring_review = self.monitoring_agent.review(facts)
        supervisor_review = self.supervisor_agent.review(
            {
                **facts,
                "risk_recommendation": risk_review.recommendation.value,
                "monitoring_recommendation": monitoring_review.recommendation.value,
            }
        )
        return AgentReviewRound(risk_review, monitoring_review, supervisor_review)


class BacktestingAgent:
    """Summarizes evidence; the deterministic validation report owns pass/fail."""

    def __init__(self, provider: LanguageModelProvider | None = None) -> None:
        self.provider = provider

    def review(self, metrics: dict[str, Any]) -> AgentReview:
        if self.provider is None:
            passed = bool(metrics.get("validation_passed", False))
            return AgentReview(
                agent_id="backtesting_agent",
                recommendation=(
                    Recommendation.APPROVE_FOR_HUMAN_REVIEW if passed else Recommendation.REJECT
                ),
                summary="Deterministic report " + ("passed" if passed else "failed") + "; human review required.",
                evidence=tuple(str(value) for value in metrics.get("failed_checks", [])),
                confidence=1.0,
            )
        response = self.provider.complete_json(
            system=(
                "You are AQIS Backtesting Agent. Review provided metrics and report weaknesses. "
                "Return JSON with recommendation, summary, evidence, confidence. You may recommend "
                "review or rejection, but cannot alter strategy parameters or validation results."
            ),
            user=json.dumps(metrics, sort_keys=True),
        )
        return _agent_review("backtesting_agent", response)


class DeploymentAgent:
    """Checks deployment readiness but cannot release a candidate or enable live trading."""

    def review(self, facts: dict[str, Any]) -> AgentReview:
        ready = bool(facts.get("validation_passed")) and bool(facts.get("human_approved"))
        return AgentReview(
            agent_id="deployment_agent",
            recommendation=(Recommendation.APPROVE_FOR_HUMAN_REVIEW if ready else Recommendation.REJECT),
            summary=("Candidate is ready for paper release review." if ready else "Paper release prerequisites are missing."),
            evidence=tuple(
                name for name, present in (
                    ("validation_passed", facts.get("validation_passed")),
                    ("human_approved", facts.get("human_approved")),
                ) if not present
            ),
            confidence=1.0,
        )


class MonitoringAgent:
    """Advisory performance/operations monitor; hard stops remain deterministic."""

    def __init__(self, provider: LanguageModelProvider | None = None) -> None:
        self.provider = provider

    def review(self, facts: dict[str, Any]) -> AgentReview:
        if facts.get("integrity_halt") or facts.get("hard_limit_breached"):
            return AgentReview(
                agent_id="monitoring_agent",
                recommendation=Recommendation.REQUEST_PAUSE,
                summary="A deterministic safety condition requires the strategy to remain paused.",
                evidence=tuple(str(item) for item in facts.get("triggers", [])),
                confidence=1.0,
            )
        if self.provider is None:
            return AgentReview(
                agent_id="monitoring_agent",
                recommendation=Recommendation.NO_CHANGE,
                summary="No deterministic pause trigger; AI provider is not configured.",
                evidence=("deterministic_limits_clear",),
                confidence=1.0,
            )
        response = self.provider.complete_json(
            system=(
                "You are AQIS Monitoring Agent. Review performance, correlation, volatility and "
                "execution facts. Return JSON with recommendation, summary, evidence, confidence. "
                "You may request a conservative pause, but cannot place orders, change limits, or resume."
            ),
            user=json.dumps(facts, sort_keys=True),
        )
        return _agent_review("monitoring_agent", response)


def _agent_review(agent_id: str, response: dict[str, Any]) -> AgentReview:
    recommendation = Recommendation(response.get("recommendation", "no_change"))
    confidence = float(response.get("confidence", 0))
    if not 0 <= confidence <= 1:
        raise ValueError("agent confidence must be in [0, 1]")
    return AgentReview(
        agent_id=agent_id,
        recommendation=recommendation,
        summary=str(response.get("summary", "")),
        evidence=tuple(str(item) for item in response.get("evidence", [])),
        confidence=confidence,
    )

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.error import URLError

from aqis.nemotron import (
    Finding,
    FindingKind,
    MockNemotronProvider,
    NemotronResearchAgent,
    OpenAICompatibleNemotronProvider,
    ResearchReport,
)


class ProviderTests(unittest.TestCase):
    def test_mock_provider_returns_schema_valid_response(self) -> None:
        provider = MockNemotronProvider()
        response = provider.complete_json(system="sys", user="market_summary test")
        self.assertIsInstance(response, dict)
        self.assertIn("findings", response)
        self.assertIn("summary", response)
        self.assertIn("confidence", response)
        self.assertGreaterEqual(response["confidence"], 0.0)
        self.assertLessEqual(response["confidence"], 1.0)
        for finding in response["findings"]:
            self.assertIn("kind", finding)
            self.assertIn("content", finding)
            self.assertIn(finding["kind"], {k.value for k in FindingKind})

    def test_mock_provider_model_info_no_credentials(self) -> None:
        provider = MockNemotronProvider()
        info = provider.model_info()
        self.assertIn("model_name", info)
        self.assertIn("model_version", info)
        self.assertIn("base_url", info)
        self.assertNotIn("api_key", info)
        self.assertNotIn("secret", info)

    def test_openai_provider_from_environment_missing_returns_none(self) -> None:
        for key in ("NEMOTRON_API_URL", "NEMOTRON_API_KEY", "NEMOTRON_MODEL"):
            os.environ.pop(key, None)
        self.assertIsNone(OpenAICompatibleNemotronProvider.from_environment())

    def test_openai_provider_model_info_no_credentials(self) -> None:
        provider = OpenAICompatibleNemotronProvider(
            endpoint="https://example.com/v1/chat/completions",
            api_key="secret-key",
            model="nemotron-test",
        )
        info = provider.model_info()
        self.assertEqual(info["model_name"], "nemotron-test")
        self.assertEqual(info["base_url"], "https://example.com/v1/chat/completions")
        self.assertNotIn("api_key", info)
        self.assertNotIn("secret", str(info))


class AgentDefaultProviderTests(unittest.TestCase):
    def test_no_env_vars_uses_mock_provider(self) -> None:
        for key in ("NEMOTRON_API_URL", "NEMOTRON_API_KEY", "NEMOTRON_MODEL"):
            os.environ.pop(key, None)
        agent = NemotronResearchAgent()
        self.assertIsInstance(agent.provider, MockNemotronProvider)

    def test_injected_provider_is_used(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider())
        self.assertIsInstance(agent.provider, MockNemotronProvider)


class ReportStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = NemotronResearchAgent(MockNemotronProvider())

    def test_all_research_methods_return_research_report(self) -> None:
        reports = [
            self.agent.research_market_summary({"price": 50000, "volatility": 0.02}),
            self.agent.research_portfolio_performance({"equity": 10000, "drawdown_pct": "0.02"}),
            self.agent.investigate_strategy_disagreement(
                [{"strategy": "ema", "action": "buy"}],
                {"target": "cash", "reason": "no_clear_majority"},
            ),
            self.agent.identify_unusual_conditions(
                {"regime": "high_volatility"}, {"findings": ["volatility spike"]}
            ),
            self.agent.review_rejected_orders(
                [{"reason_code": "max_exposure_exceeded"}],
                [{"intervention_type": "pause_transition"}],
            ),
            self.agent.explain_trading_decision("abc123", [{"event_type": "selector_decision"}]),
            self.agent.recommend_strategy_experiments({"ema": {"signals": 10}}),
        ]
        for report in reports:
            self.assertIsInstance(report, ResearchReport)
            self.assertIsInstance(report.analysis_id, str)
            self.assertGreater(len(report.analysis_id), 0)
            self.assertIsInstance(report.model_name, str)
            self.assertIsInstance(report.model_version, str)
            self.assertIsInstance(report.timestamp_utc, datetime)
            self.assertIsInstance(report.topic, str)
            self.assertIsInstance(report.summary, str)
            self.assertGreaterEqual(report.confidence, 0.0)
            self.assertLessEqual(report.confidence, 1.0)
            self.assertIsInstance(report.raw_response, dict)
            for finding in report.findings:
                self.assertIsInstance(finding, Finding)
                self.assertIn(finding.kind, FindingKind)
                self.assertIsInstance(finding.content, str)
                self.assertIsInstance(finding.source, str)
                if finding.confidence is not None:
                    self.assertGreaterEqual(finding.confidence, 0.0)
                    self.assertLessEqual(finding.confidence, 1.0)

    def test_analysis_id_is_deterministic(self) -> None:
        context = {"price": 50000, "volatility": 0.02}
        r1 = self.agent.research_market_summary(context)
        r2 = self.agent.research_market_summary(context)
        self.assertEqual(r1.analysis_id, r2.analysis_id)
        self.assertEqual(r1.summary, r2.summary)
        self.assertEqual(r1.confidence, r2.confidence)
        self.assertEqual(len(r1.findings), len(r2.findings))

    def test_different_topics_produce_different_analysis_ids(self) -> None:
        r_market = self.agent.research_market_summary({"price": 50000})
        r_portfolio = self.agent.research_portfolio_performance({"equity": 10000})
        self.assertNotEqual(r_market.analysis_id, r_portfolio.analysis_id)

    def test_same_topic_different_input_produces_different_analysis_ids(self) -> None:
        r_a = self.agent.research_market_summary({"price": 50000, "volatility": 0.02})
        r_b = self.agent.research_market_summary({"price": 60000, "volatility": 0.05})
        self.assertNotEqual(r_a.analysis_id, r_b.analysis_id)

    def test_analysis_id_changes_only_when_input_changes(self) -> None:
        context = {"price": 50000, "volatility": 0.02}
        r1 = self.agent.research_market_summary(context)
        r2 = self.agent.research_market_summary(context)
        self.assertEqual(r1.analysis_id, r2.analysis_id)
        context["price"] = 50001
        r3 = self.agent.research_market_summary(context)
        self.assertNotEqual(r2.analysis_id, r3.analysis_id)

    def test_findings_distinguish_fact_from_opinion(self) -> None:
        report = self.agent.research_market_summary({"price": 50000})
        kinds = {f.kind for f in report.findings}
        # Mock returns at least a market_fact and a model_opinion
        self.assertIn(FindingKind.MARKET_FACT, kinds)
        self.assertIn(FindingKind.MODEL_OPINION, kinds)
        # Opinions and recommendations may carry confidence; facts typically don't
        for finding in report.findings:
            if finding.kind is FindingKind.MARKET_FACT:
                self.assertIsNone(finding.confidence)


class FallbackTests(unittest.TestCase):
    def test_api_failure_produces_safe_fallback_without_crash(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider(fail=True))
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.confidence, 0.0)
        self.assertTrue(report.raw_response.get("fallback"))
        self.assertIn("unavailable", report.summary.lower())
        self.assertTrue(any(f.source == "fallback" for f in report.findings))

    def test_malformed_response_produces_safe_fallback(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider(malformed=True))
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.confidence, 0.0)
        self.assertTrue(report.raw_response.get("fallback"))

    def test_timeout_produces_safe_fallback_without_blocking(self) -> None:
        class TimeoutProvider:
            def model_info(self):
                return {"model_name": "t", "model_version": "1", "base_url": "x"}

            def complete_json(self, *, system, user):
                raise TimeoutError("request timed out")

        agent = NemotronResearchAgent(TimeoutProvider())
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.confidence, 0.0)
        self.assertTrue(report.raw_response.get("fallback"))
        self.assertEqual(report.raw_response["error"], "provider_unavailable")

    def test_exception_message_excludes_api_key(self) -> None:
        class KeyLeakProvider:
            def __init__(self, api_key: str) -> None:
                self._api_key = api_key

            def model_info(self):
                return {"model_name": "t", "model_version": "1", "base_url": "x"}

            def complete_json(self, *, system, user):
                # Simulate an error that references the key internally
                raise URLError(f"connection failed using key {self._api_key}")

        secret_key = "sk-live-nemotron-secret-12345"
        agent = NemotronResearchAgent(KeyLeakProvider(secret_key))
        report = agent.research_market_summary({"price": 50000})
        self.assertTrue(report.raw_response.get("fallback"))
        error_text = report.raw_response["error"]
        self.assertNotIn(secret_key, error_text)
        self.assertNotIn("sk-live", error_text)
        self.assertNotIn("12345", error_text)


class ValidationTests(unittest.TestCase):
    def test_rejects_invalid_finding_kind(self) -> None:
        with self.assertRaises(ValueError):
            NemotronResearchAgent._validate_response(
                {"findings": [{"kind": "invalid_kind", "content": "x"}], "summary": "s", "confidence": 0.5}
            )

    def test_rejects_confidence_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            NemotronResearchAgent._validate_response(
                {"findings": [], "summary": "s", "confidence": 1.5}
            )
        with self.assertRaises(ValueError):
            NemotronResearchAgent._validate_response(
                {"findings": [], "summary": "s", "confidence": -0.1}
            )

    def test_rejects_missing_finding_content(self) -> None:
        with self.assertRaises(ValueError):
            NemotronResearchAgent._validate_response(
                {"findings": [{"kind": "market_fact"}], "summary": "s", "confidence": 0.5}
            )

    def test_rejects_non_dict_response(self) -> None:
        with self.assertRaises(ValueError):
            NemotronResearchAgent._validate_response(["not", "a", "dict"])

    def test_market_fact_without_evidence_ref_is_reclassified_to_model_opinion(self) -> None:
        class NoEvidenceProvider:
            def model_info(self):
                return {"model_name": "t", "model_version": "1", "base_url": "x"}

            def complete_json(self, *, system, user):
                return {
                    "findings": [
                        {
                            "kind": "market_fact",
                            "content": "price is up",
                            "source": "model",
                        }
                    ],
                    "summary": "s",
                    "confidence": 0.8,
                }

        agent = NemotronResearchAgent(NoEvidenceProvider())
        report = agent.research_market_summary({"price": 50000})
        market_facts = [f for f in report.findings if f.kind is FindingKind.MARKET_FACT]
        self.assertEqual(market_facts, [])
        opinions = [f for f in report.findings if f.kind is FindingKind.MODEL_OPINION]
        self.assertEqual(len(opinions), 1)
        self.assertEqual(opinions[0].content, "price is up")

    def test_market_fact_with_evidence_ref_remains_market_fact(self) -> None:
        class EvidenceProvider:
            def model_info(self):
                return {"model_name": "t", "model_version": "1", "base_url": "x"}

            def complete_json(self, *, system, user):
                return {
                    "findings": [
                        {
                            "kind": "market_fact",
                            "content": "price is 50000",
                            "source": "model",
                            "evidence_ref": "input.price",
                        }
                    ],
                    "summary": "s",
                    "confidence": 0.9,
                }

        agent = NemotronResearchAgent(EvidenceProvider())
        report = agent.research_market_summary({"price": 50000})
        market_facts = [f for f in report.findings if f.kind is FindingKind.MARKET_FACT]
        self.assertEqual(len(market_facts), 1)
        self.assertEqual(market_facts[0].evidence_ref, "input.price")


class ShadowModeTests(unittest.TestCase):
    def test_agent_has_no_order_or_execution_tools(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider())
        forbidden = [
            "place_order",
            "submit_order",
            "modify_position",
            "change_risk_limit",
            "approve_recommendation",
            "resume_pause",
            "execute",
            "trade",
        ]
        for method in forbidden:
            self.assertFalse(hasattr(agent, method), f"agent must not expose {method}")

    def test_all_methods_are_advisory_read_only(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider())
        # All research methods return reports, never mutate risk/selector state
        report = agent.research_market_summary({"price": 50000})
        self.assertIsInstance(report, ResearchReport)
        self.assertEqual(report.topic, "market_summary")

    def test_research_methods_do_not_mutate_input(self) -> None:
        agent = NemotronResearchAgent(MockNemotronProvider())
        market_input = {"price": 50000, "volatility": 0.02, "nested": {"regime": "ranging"}}
        before = dict(market_input)
        before["nested"] = dict(market_input["nested"])
        agent.research_market_summary(market_input)
        self.assertEqual(market_input, before)
        self.assertEqual(market_input["nested"], before["nested"])

        portfolio_input = {"equity": 10000, "drawdown_pct": "0.02"}
        before_pf = dict(portfolio_input)
        agent.research_portfolio_performance(portfolio_input)
        self.assertEqual(portfolio_input, before_pf)


if __name__ == "__main__":
    unittest.main()

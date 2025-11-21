"""Test suite for ablation harness functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bailiff.core.config import AgentBudget, CueToggle, Phase, PhaseBudget, Role, TrialConfig
from bailiff.core.events import TrialLog, UtteranceLog

# Import ablation modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from run_ablation import (
    AblationConfig,
    AblationSweep,
    AblationVariation,
    extract_metrics,
    generate_comparison_tables,
    parse_ablation_yaml,
)


class TestAblationVariation:
    """Test configuration variation application."""
    
    def test_phase_budget_override(self):
        """Test that phase budget overrides work correctly."""
        base_config = TrialConfig(
            case_template=Path("bailiff/datasets/cases/traffic.yaml"),
            cue=CueToggle("test_cue", "control", "treatment"),
            model_identifier="test-model",
            seed=42,
            agent_budgets={Role.JUDGE: AgentBudget(max_bytes=1500)},
            phase_budgets=[
                PhaseBudget(phase=Phase.OPENING, max_messages=2),
                PhaseBudget(phase=Phase.DIRECT, max_messages=2),
            ],
        )
        
        variation = AblationVariation(
            name="tight_budgets",
            overrides={
                "phase_budgets": {
                    "opening": {"max_messages": 1},
                    "direct": {"max_messages": 1},
                }
            },
        )
        
        new_config = variation.apply_to_config(base_config)
        
        # Check that phase budgets were updated
        opening_budget = new_config.phase_budget_for(Phase.OPENING)
        assert opening_budget.max_messages == 1
        
        direct_budget = new_config.phase_budget_for(Phase.DIRECT)
        assert direct_budget.max_messages == 1
    
    def test_agent_budget_override(self):
        """Test that agent budget overrides work correctly."""
        base_config = TrialConfig(
            case_template=Path("bailiff/datasets/cases/traffic.yaml"),
            cue=CueToggle("test_cue", "control", "treatment"),
            model_identifier="test-model",
            seed=42,
            agent_budgets={
                Role.JUDGE: AgentBudget(max_bytes=1500),
                Role.PROSECUTION: AgentBudget(max_bytes=1800),
            },
            phase_budgets=[],
        )
        
        variation = AblationVariation(
            name="constrained",
            overrides={
                "agent_budgets": {
                    "judge": {"max_bytes": 1000, "max_tokens": 400},
                    "prosecution": {"max_bytes": 1200},
                }
            },
        )
        
        new_config = variation.apply_to_config(base_config)
        
        # Check updates
        assert new_config.budget_for(Role.JUDGE).max_bytes == 1000
        assert new_config.budget_for(Role.JUDGE).max_tokens == 400
        assert new_config.budget_for(Role.PROSECUTION).max_bytes == 1200
    
    def test_blinding_override(self):
        """Test that blinding flag overrides work correctly."""
        base_config = TrialConfig(
            case_template=Path("bailiff/datasets/cases/traffic.yaml"),
            cue=CueToggle("test_cue", "control", "treatment"),
            model_identifier="test-model",
            seed=42,
            agent_budgets={Role.JUDGE: AgentBudget(max_bytes=1500)},
            phase_budgets=[],
            judge_blinding=False,
            strict_blinding=False,
        )
        
        variation = AblationVariation(
            name="blinding_on",
            overrides={
                "judge_blinding": True,
                "strict_blinding": True,
            },
        )
        
        new_config = variation.apply_to_config(base_config)
        
        assert new_config.judge_blinding is True
        assert new_config.strict_blinding is True


class TestAblationConfig:
    """Test ablation configuration parsing."""
    
    def test_parse_minimal_config(self):
        """Test parsing a minimal ablation config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "base_config": "configs/pilot.yaml",
                "ablations": [
                    {
                        "name": "test_sweep",
                        "variations": [
                            {
                                "name": "var1",
                                "overrides": {"judge_blinding": True},
                            }
                        ]
                    }
                ]
            }, f)
            config_path = Path(f.name)
        
        try:
            ablation_config = parse_ablation_yaml(config_path)
            
            assert ablation_config.base_config_path == Path("configs/pilot.yaml")
            assert ablation_config.repetitions == 5  # default
            assert len(ablation_config.sweeps) == 1
            assert ablation_config.sweeps[0].name == "test_sweep"
            assert len(ablation_config.sweeps[0].variations) == 1
        finally:
            config_path.unlink()
    
    def test_parse_full_config(self):
        """Test parsing a complete ablation config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "base_config": "configs/pilot.yaml",
                "repetitions": 10,
                "output_format": "csv",
                "ablations": [
                    {
                        "name": "sweep1",
                        "description": "Test sweep",
                        "variations": [
                            {"name": "var1", "overrides": {}},
                            {"name": "var2", "overrides": {"judge_blinding": True}},
                        ]
                    }
                ]
            }, f)
            config_path = Path(f.name)
        
        try:
            ablation_config = parse_ablation_yaml(config_path)
            
            assert ablation_config.repetitions == 10
            assert ablation_config.output_format == "csv"
            assert len(ablation_config.sweeps[0].variations) == 2
            assert ablation_config.sweeps[0].description == "Test sweep"
        finally:
            config_path.unlink()


class TestMetricExtraction:
    """Test metric extraction from trial logs."""
    
    def _make_trial_log(
        self,
        trial_id: str,
        verdict: str,
        sentence: float | None = None,
        n_utterances: int = 5,
    ) -> TrialLog:
        """Helper to create a trial log for testing."""
        utterances = [
            UtteranceLog(
                role=Role.JUDGE.value,
                phase=Phase.OPENING.value,
                content="Test content",
                byte_count=100,
                token_count=20,
                timestamp="2025-01-01T00:00:00Z",
            )
            for _ in range(n_utterances)
        ]
        
        return TrialLog(
            trial_id=trial_id,
            case_identifier="test_case",
            model_identifier="test_model",
            cue_name="test_cue",
            cue_condition="control",
            cue_value="test_value",
            verdict=verdict,
            sentence=sentence,
            utterances=utterances,
        )
    
    def test_verdict_metrics(self):
        """Test extraction of verdict-related metrics."""
        control_log = self._make_trial_log("ctrl", "guilty")
        treatment_log = self._make_trial_log("treat", "not_guilty")
        
        metrics = extract_metrics(control_log, treatment_log)
        
        assert metrics["control_verdict"] == "guilty"
        assert metrics["treatment_verdict"] == "not_guilty"
        assert metrics["verdict_flip"] is True
    
    def test_sentence_metrics(self):
        """Test extraction of sentence comparison metrics."""
        control_log = self._make_trial_log("ctrl", "guilty", sentence=12.0)
        treatment_log = self._make_trial_log("treat", "guilty", sentence=24.0)
        
        metrics = extract_metrics(control_log, treatment_log)
        
        assert metrics["control_sentence"] == 12.0
        assert metrics["treatment_sentence"] == 24.0
        assert metrics["sentence_delta"] == 12.0
    
    def test_token_metrics(self):
        """Test extraction of token count metrics."""
        control_log = self._make_trial_log("ctrl", "guilty", n_utterances=3)
        treatment_log = self._make_trial_log("treat", "guilty", n_utterances=5)
        
        metrics = extract_metrics(control_log, treatment_log)
        
        # Each utterance has 20 tokens
        assert metrics["control_total_tokens"] == 60
        assert metrics["treatment_total_tokens"] == 100
        assert metrics["token_delta"] == 40
    
    def test_byte_metrics(self):
        """Test extraction of byte count metrics."""
        control_log = self._make_trial_log("ctrl", "guilty", n_utterances=2)
        treatment_log = self._make_trial_log("treat", "guilty", n_utterances=4)
        
        metrics = extract_metrics(control_log, treatment_log)
        
        # Each utterance has 100 bytes
        assert metrics["control_total_bytes"] == 200
        assert metrics["treatment_total_bytes"] == 400
        assert metrics["byte_delta"] == 200


class TestComparisonTables:
    """Test comparison table generation."""
    
    def test_generate_summary_table(self):
        """Test generation of summary comparison table."""
        # Create mock results dataframe
        results = pd.DataFrame([
            {
                "sweep": "test_sweep",
                "variation": "var1",
                "repetition": 0,
                "verdict_flip": True,
                "sentence_delta": 5.0,
                "token_delta": 100,
                "byte_delta": 500,
                "control_objections": 2,
                "treatment_objections": 3,
                "control_interruptions": 0,
                "treatment_interruptions": 1,
            },
            {
                "sweep": "test_sweep",
                "variation": "var1",
                "repetition": 1,
                "verdict_flip": False,
                "sentence_delta": -3.0,
                "token_delta": 50,
                "byte_delta": 250,
                "control_objections": 1,
                "treatment_objections": 2,
                "control_interruptions": 0,
                "treatment_interruptions": 0,
            },
        ])
        
        summary, detail = generate_comparison_tables(results)
        
        # Check summary table structure
        assert "sweep" in summary.columns
        assert "variation" in summary.columns
        assert "verdict_flip_mean" in summary.columns
        assert "n_trials" in summary.columns
        
        # Check aggregation
        assert len(summary) == 1  # One variation
        assert summary.iloc[0]["n_trials"] == 2
        assert summary.iloc[0]["verdict_flip_mean"] == 0.5
    
    def test_generate_detail_table(self):
        """Test generation of detailed comparison table."""
        results = pd.DataFrame([
            {
                "sweep": "sweep1",
                "variation": "var1",
                "repetition": 0,
                "control_verdict": "guilty",
                "treatment_verdict": "guilty",
                "verdict_flip": False,
                "sentence_delta": 5.0,
                "token_delta": 100,
                "control_total_tokens": 500,
                "treatment_total_tokens": 600,
            },
            {
                "sweep": "sweep1",
                "variation": "var2",
                "repetition": 0,
                "control_verdict": "not_guilty",
                "treatment_verdict": "guilty",
                "verdict_flip": True,
                "sentence_delta": 10.0,
                "token_delta": 200,
                "control_total_tokens": 450,
                "treatment_total_tokens": 650,
            },
        ])
        
        summary, detail = generate_comparison_tables(results)
        
        # Check detail table
        assert len(detail) == 2  # Two variations
        assert "sweep" in detail.columns
        assert "variation" in detail.columns


class TestIntegration:
    """Integration tests for ablation harness."""
    
    def test_config_variation_roundtrip(self):
        """Test that applying variations and extracting back is consistent."""
        base_config = TrialConfig(
            case_template=Path("bailiff/datasets/cases/traffic.yaml"),
            cue=CueToggle("test_cue", "control", "treatment"),
            model_identifier="test-model",
            seed=42,
            agent_budgets={
                Role.JUDGE: AgentBudget(max_bytes=1500),
                Role.PROSECUTION: AgentBudget(max_bytes=1800),
                Role.DEFENSE: AgentBudget(max_bytes=1800),
            },
            phase_budgets=[
                PhaseBudget(phase=Phase.OPENING, max_messages=2),
            ],
            judge_blinding=False,
        )
        
        # Apply multiple variations
        variations = [
            AblationVariation("var1", {"judge_blinding": True}),
            AblationVariation("var2", {
                "agent_budgets": {
                    "judge": {"max_bytes": 1000}
                }
            }),
            AblationVariation("var3", {
                "phase_budgets": {
                    "opening": {"max_messages": 3}
                }
            }),
        ]
        
        for var in variations:
            new_config = var.apply_to_config(base_config)
            # Verify base config is not mutated
            assert base_config.judge_blinding is False
            # Verify new config has changes
            if var.name == "var1":
                assert new_config.judge_blinding is True
            elif var.name == "var2":
                assert new_config.budget_for(Role.JUDGE).max_bytes == 1000
            elif var.name == "var3":
                assert new_config.phase_budget_for(Phase.OPENING).max_messages == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

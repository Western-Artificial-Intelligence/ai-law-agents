# Configuration Files

This directory contains YAML configuration files for different B.A.I.L.I.F.F. workflows.

## Files

- **`pilot.yaml`**: Single paired trial configuration for quick testing
- **`batch.yaml`**: Multi-case/multi-model matrix configuration
- **`ablation_example.yaml`**: Example ablation study with 5 sweep types
- **`fir_batch.yaml`**: FIR/Compute-Alliance batch configuration for paper-scale runs
- **`nibi_local_*`**: Nibi local-GPU model panel configs (full/reduced tiers)
- **`nibi_local_*_reduced.yaml`**: Nibi local-GPU add-on configs for larger SOTA models
- **`nibi_local_*_research.yaml`**: Nibi research-backed multi-cue panel configs
- **`nibi_local_*_us_subset.yaml`**: US-law robustness subset configs (external validity checks)

## Quick Reference

### Pilot Config (`pilot.yaml`)

Run a single paired trial:

```bash
python scripts/run_pilot_trial.py --config configs/pilot.yaml --backend echo
```

### Ablation Config (`ablation_example.yaml`)

Run systematic configuration sweeps:

```bash
python scripts/run_ablation.py --config configs/ablation_example.yaml --backend echo
```

## Ablation Configuration Format

```yaml
base_config: configs/pilot.yaml # Base configuration to vary
repetitions: 5 # Trials per variation
output_format: both # 'csv', 'markdown', or 'both'

ablations:
  - name: sweep_name
    description: "What you're testing"
    variations:
      - name: variation_1
        overrides:
          judge_blinding: true
          agent_budgets:
            judge: { max_bytes: 1000 }
```

## Common Override Patterns

### Phase Budgets

```yaml
overrides:
  phase_budgets:
    opening: { max_messages: 1 }
    direct: { max_messages: 2 }
```

### Agent Budgets

```yaml
overrides:
  agent_budgets:
    judge: { max_bytes: 1500, max_tokens: 600 }
    prosecution: { max_bytes: 1800 }
```

### Blinding Modes

```yaml
overrides:
  judge_blinding: true
  strict_blinding: true
```

## See Also

- Full ablation guide: `docs/ABLATION_GUIDE.md`
- User guide: `docs/USER_GUIDE.md`
- Example ablation: `configs/ablation_example.yaml`
- Nibi 6-model panel: `docs/NIBI_6_MODEL_PANEL.md`
- Nibi SOTA add-ons: `docs/NIBI_SOTA_ADDONS.md`
- Nibi research panel: `docs/NIBI_RESEARCH_PANEL.md`
- Research cue matrix: `docs/RESEARCH_CUE_MATRIX.md`
- US robustness launcher: `scripts/submit_nibi_us_robustness_subset.sh`

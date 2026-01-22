# Configuration Files

This directory contains YAML configuration files for different B.A.I.L.I.F.F. workflows.

## Files

- **`pilot.yaml`**: Single paired trial configuration for quick testing
- **`batch.yaml`**: Multi-case/multi-model matrix configuration
- **`ablation_example.yaml`**: Example ablation study with 5 sweep types

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

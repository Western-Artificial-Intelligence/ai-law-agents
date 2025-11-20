# Ablation Study Guide

This guide explains how to use the B.A.I.L.I.F.F. ablation harness to systematically evaluate the impact of configuration choices on trial outcomes and procedural metrics.

## Overview

An **ablation study** removes or varies components of a system to understand their individual contributions. In the context of B.A.I.L.I.F.F., ablation studies help answer questions like:

- Do tighter phase message budgets reduce verdict bias?
- Does judge blinding effectively prevent cue leakage?
- How do agent byte limits affect procedural fairness metrics?
- What's the interaction between budget constraints and blinding modes?

The ablation harness automates running multiple trial configurations, comparing results, and generating analysis-ready reports.

## Quick Start

### 1. Create an Ablation Configuration

Create a YAML file (e.g., `configs/my_ablation.yaml`) defining your experimental sweeps:

```yaml
base_config: configs/pilot.yaml
repetitions: 5
output_format: both

ablations:
  - name: blinding_test
    description: "Test judge blinding effectiveness"
    variations:
      - name: no_blinding
        overrides:
          judge_blinding: false

      - name: with_blinding
        overrides:
          judge_blinding: true
```

### 2. Run the Ablation Study

```bash
python scripts/run_ablation.py \
  --config configs/my_ablation.yaml \
  --out runs/ablation_logs.jsonl \
  --comparison-csv runs/comparison.csv \
  --comparison-md runs/comparison.md \
  --backend echo
```

For real LLM backends:

```bash
# Using Groq
python scripts/run_ablation.py \
  --config configs/my_ablation.yaml \
  --backend groq \
  --model llama3-8b-8192 \
  --out runs/ablation_logs.jsonl

# Using Gemini
python scripts/run_ablation.py \
  --config configs/my_ablation.yaml \
  --backend gemini \
  --model gemini-1.5-flash \
  --out runs/ablation_logs.jsonl
```

### 3. Review Results

The harness generates:

- **JSONL logs** (`--out`): Full trial logs for all variations and repetitions
- **CSV comparison** (`--comparison-csv`): Machine-readable metrics table
- **Markdown report** (`--comparison-md`): Human-readable formatted results
- **Console summary**: Aggregated statistics printed to terminal

## Configuration Format

### Top-Level Structure

```yaml
# Path to base configuration (required)
base_config: configs/pilot.yaml

# Number of trial pairs per variation (default: 5)
repetitions: 10

# Output format: 'csv', 'markdown', or 'both' (default: 'both')
output_format: both

# List of ablation sweeps (required)
ablations:
  - name: sweep_name_1
    description: "Optional description"
    variations: [...]

  - name: sweep_name_2
    variations: [...]
```

### Variation Overrides

Each variation can override any of the following configuration parameters:

#### Phase Budgets

```yaml
overrides:
  phase_budgets:
    opening:
      max_messages: 3
      allow_interruptions: true
    direct:
      max_messages: 2
    cross:
      max_messages: 2
    redirect:
      max_messages: 1
    closing:
      max_messages: 2
```

#### Agent Budgets

```yaml
overrides:
  agent_budgets:
    judge:
      max_bytes: 2000
      max_tokens: 800
    prosecution:
      max_bytes: 2500
      max_tokens: 1000
    defense:
      max_bytes: 2500
      max_tokens: 1000
```

#### Blinding Modes

```yaml
overrides:
  judge_blinding: true # Hide cue value from judge
  strict_blinding: true # Hide BOTH control and treatment values
```

#### Policy Enforcement

```yaml
overrides:
  enforce_role_phase_policy: true # Strict phase-role validation
```

#### Combined Overrides

You can combine multiple overrides in a single variation:

```yaml
- name: strict_constrained
  overrides:
    judge_blinding: true
    enforce_role_phase_policy: true
    agent_budgets:
      judge:
        max_bytes: 1000
    phase_budgets:
      opening:
        max_messages: 1
```

## Example Ablation Designs

### Budget Sensitivity Analysis

Test how budget constraints affect outcomes:

```yaml
ablations:
  - name: budget_sweep
    variations:
      - name: tight
        overrides:
          agent_budgets:
            judge: { max_bytes: 1000 }
            prosecution: { max_bytes: 1200 }
            defense: { max_bytes: 1200 }

      - name: standard
        overrides:
          agent_budgets:
            judge: { max_bytes: 1500 }
            prosecution: { max_bytes: 1800 }
            defense: { max_bytes: 1800 }

      - name: generous
        overrides:
          agent_budgets:
            judge: { max_bytes: 2500 }
            prosecution: { max_bytes: 3000 }
            defense: { max_bytes: 3000 }
```

### Blinding Efficacy Test

Evaluate different blinding strategies:

```yaml
ablations:
  - name: blinding_modes
    variations:
      - name: no_blind
        overrides:
          judge_blinding: false
          strict_blinding: false

      - name: standard_blind
        overrides:
          judge_blinding: true
          strict_blinding: false

      - name: strict_blind
        overrides:
          judge_blinding: true
          strict_blinding: true
```

### Interaction Effects

Test how multiple factors interact:

```yaml
ablations:
  - name: budget_blind_interaction
    variations:
      - name: tight_no_blind
        overrides:
          judge_blinding: false
          agent_budgets:
            judge: { max_bytes: 1000 }

      - name: tight_with_blind
        overrides:
          judge_blinding: true
          agent_budgets:
            judge: { max_bytes: 1000 }

      - name: loose_no_blind
        overrides:
          judge_blinding: false
          agent_budgets:
            judge: { max_bytes: 2500 }

      - name: loose_with_blind
        overrides:
          judge_blinding: true
          agent_budgets:
            judge: { max_bytes: 2500 }
```

### Phase Control Ablation

Isolate the effect of specific trial phases:

```yaml
ablations:
  - name: phase_control
    variations:
      - name: skip_redirect
        overrides:
          phase_budgets:
            redirect: { max_messages: 0 }

      - name: extended_cross
        overrides:
          phase_budgets:
            cross: { max_messages: 5 }

      - name: brief_opening
        overrides:
          phase_budgets:
            opening: { max_messages: 1 }
```

## Output Metrics

The ablation harness computes the following comparison metrics:

### Outcome Metrics

- **verdict_flip_mean**: Proportion of pairs with divergent verdicts
- **verdict_flip_sum**: Total number of verdict flips
- **control_verdict**: Guilty verdicts in control condition
- **treatment_verdict**: Guilty verdicts in treatment condition

### Sentence Metrics

- **sentence_delta_mean**: Average sentence difference (treatment - control)
- **sentence_delta_std**: Standard deviation of sentence differences
- **sentence_delta_min/max**: Range of sentence differences

### Token/Byte Metrics

- **token_delta_mean**: Average token count difference
- **token_delta_std**: Standard deviation of token differences
- **byte_delta_mean**: Average byte count difference
- **control_total_tokens**: Mean tokens in control trials
- **treatment_total_tokens**: Mean tokens in treatment trials

### Procedural Metrics

- **control_objections**: Mean objections in control trials
- **treatment_objections**: Mean objections in treatment trials
- **control_interruptions**: Mean interruptions in control trials
- **treatment_interruptions**: Mean interruptions in treatment trials

### Role-Specific Counts

- **control_judge_utterances**: Judge utterance count (control)
- **treatment_judge_utterances**: Judge utterance count (treatment)
- Similar metrics for prosecution and defense

## Interpreting Results

### Reading the Summary Table

The summary table aggregates metrics across repetitions:

```
| sweep          | variation      | verdict_flip_mean | sentence_delta_mean | n_trials |
|----------------|----------------|-------------------|---------------------|----------|
| blinding_test  | no_blinding    | 0.40              | 5.2                 | 5        |
| blinding_test  | with_blinding  | 0.20              | 1.8                 | 5        |
```

**Interpretation**: Blinding reduced verdict flips from 40% to 20% and decreased average sentence disparity from 5.2 to 1.8 months.

### Statistical Significance

For robust conclusions:

1. **Increase repetitions**: Use `repetitions: 20` or higher for statistical power
2. **Test multiple cases**: Modify base config to sweep across different case templates
3. **Use bootstrapping**: Export JSONL logs and apply `scripts/outcome_wild_bootstrap.py`
4. **Control for multiplicity**: Apply FDR correction when testing many variations

### Common Patterns

**Budget reduction increases variance**: Tighter budgets often lead to more variable outcomes due to truncation effects.

**Blinding effectiveness varies by cue**: Blinding may be more effective for name-based cues than dialect cues if agents use writing style as a proxy.

**Interaction effects dominate**: The combination of blinding + tight budgets may have non-additive effects.

## Advanced Usage

### Custom Prompt Variations

To test different prompts, first create prompt files:

```bash
# Create alternative judge prompt
cat > prompts/neutral_judge.txt << EOF
You are a neutral arbiter. Evaluate only the facts presented.
Avoid assumptions based on defendant characteristics.
EOF
```

Then reference in your ablation config:

```yaml
overrides:
  prompt_overrides:
    judge: prompts/neutral_judge.txt
```

**Note**: This requires extending `AblationVariation.apply_to_config()` to handle prompt file loading.

### Ablating Case Parameters

To vary the case itself:

```yaml
ablations:
  - name: case_complexity
    variations:
      - name: simple_case
        overrides:
          case_template: bailiff/datasets/cases/traffic.yaml

      - name: complex_case
        overrides:
          case_template: bailiff/datasets/cases/dui.yaml
```

### Backend-Specific Ablations

Test how model choice affects bias:

```bash
# Run with different models
python scripts/run_ablation.py \
  --config configs/ablation.yaml \
  --backend groq \
  --model llama3-8b-8192

python scripts/run_ablation.py \
  --config configs/ablation.yaml \
  --backend groq \
  --model mixtral-8x7b-32768
```

Then compare outputs across models.

## Best Practices

### Design Principles

1. **One factor at a time**: Start with simple sweeps varying a single parameter
2. **Control group**: Include a baseline variation matching the standard config
3. **Adequate repetitions**: Use ≥10 repetitions for stable estimates
4. **Document hypotheses**: Use `description` fields to record what you're testing

### Computational Efficiency

- **Start with echo backend**: Validate your config structure before using LLMs
- **Use local models**: Consider `--backend local` with quantized models for cost-effective exploration
- **Batch scheduling**: Run expensive ablations overnight or on compute clusters
- **Resume capability**: The JSONL output format supports incremental results

### Reporting Results

Include in your ablation report:

1. **Motivation**: Why these variations matter
2. **Configuration**: Base config and all variation overrides
3. **Sample size**: Number of repetitions per variation
4. **Summary statistics**: Mean and variance for key metrics
5. **Visualizations**: Plot verdict flip rates, sentence deltas
6. **Limitations**: Note any technical issues or boundary conditions

## Troubleshooting

### "No trial pairs generated"

**Cause**: Base config missing required fields.

**Fix**: Ensure `base_config` points to a valid pilot config with all required fields (case_template, cue, agent_budgets, phase_budgets).

### "Insufficient discordant pairs for McNemar estimate"

**Cause**: All pairs have identical verdicts.

**Fix**: Increase `repetitions` or check that cues are actually being toggled.

### High variance in results

**Cause**: Low repetition count or high model temperature.

**Fix**: Increase `repetitions` to 20+, or add `backend_params: {temperature: 0.1}` to base config.

### OOM errors with large ablations

**Cause**: Too many trials in memory.

**Fix**: Process results incrementally by parsing the JSONL output in chunks:

```python
import pandas as pd
from bailiff.core.io import read_jsonl

logs = []
for chunk in pd.read_json("runs/ablation_logs.jsonl", lines=True, chunksize=100):
    # Process chunk
    pass
```

## Example Workflow

### Full Ablation Pipeline

```bash
# 1. Create ablation config
cat > configs/my_study.yaml << EOF
base_config: configs/pilot.yaml
repetitions: 10
ablations:
  - name: budget_test
    variations:
      - name: baseline
        overrides: {}
      - name: tight
        overrides:
          agent_budgets:
            judge: {max_bytes: 1000}
EOF

# 2. Run with echo backend first (fast validation)
python scripts/run_ablation.py \
  --config configs/my_study.yaml \
  --backend echo \
  --out runs/ablation_echo.jsonl \
  --comparison-md runs/echo_results.md

# 3. Review echo results
cat runs/echo_results.md

# 4. Run with real LLM backend
python scripts/run_ablation.py \
  --config configs/my_study.yaml \
  --backend groq \
  --model llama3-8b-8192 \
  --out runs/ablation_groq.jsonl \
  --comparison-csv runs/groq_results.csv

# 5. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('runs/groq_results.csv')
print(df[['variation', 'verdict_flip_mean', 'sentence_delta_mean']])
"

# 6. Generate plots (optional)
python scripts/plot_ablation_results.py \
  --csv runs/groq_results.csv \
  --out plots/ablation_viz.png
```

## Integration with Analysis Pipeline

Ablation outputs integrate seamlessly with existing analysis tools:

```python
from bailiff.core.io import read_jsonl
from bailiff.metrics.outcome import mcnemar_log_odds, PairedOutcome

# Load ablation logs
logs = read_jsonl("runs/ablation_logs.jsonl")

# Filter by variation
baseline_logs = [log for log in logs if "baseline" in log.notes]
tight_logs = [log for log in logs if "tight" in log.notes]

# Compute paired outcomes
baseline_pairs = []
for i in range(0, len(baseline_logs), 2):
    baseline_pairs.append(PairedOutcome(
        control=1 if baseline_logs[i].verdict == "guilty" else 0,
        treatment=1 if baseline_logs[i+1].verdict == "guilty" else 0,
    ))

# Estimate bias
log_odds, se = mcnemar_log_odds(baseline_pairs)
print(f"Baseline log-odds: {log_odds:.3f} (SE: {se:.3f})")
```

## FAQ

**Q: Can I ablate prompts directly in the YAML?**

A: Currently, prompts are loaded from `bailiff/agents/prompts.py`. To test prompt variations, you'd need to either:

- Create multiple base configs with different prompt modules
- Extend the ablation harness to accept inline prompt overrides

**Q: How do I test across multiple cases?**

A: Create separate ablation configs for each case template, or extend the harness to sweep `case_template` as a variation.

**Q: Can I use ablations for placebo/negative controls?**

A: Yes! Set up a sweep with placebo cues:

```yaml
overrides:
  cue: name_placebo # Requires extending config loader to accept cue overrides
```

**Q: How do I export results to R for advanced stats?**

A: Use `--comparison-csv` and load in R:

```r
library(tidyverse)
results <- read_csv("runs/groq_results.csv")
model <- lmer(verdict_flip ~ variation + (1|repetition), data=results)
```

## See Also

- [User Guide](USER_GUIDE.md): Base trial configuration and metrics
- [API Reference](API.md): Core modules and data structures
- [Design Overview](../DESIGN.md): Architecture and data flow
- [Outcome Scripts](OUTCOME_SCRIPTS.md): Statistical analysis methods

# Groq + GAUL Integration Tickets

## Ticket 1: Configure Groq API Keys on GAUL Server

**Priority:** High
**Labels:** `infrastructure`, `gaul`, `groq`
**Estimate:** 1 point

### Description
Update the GAUL setup script to support Groq API key configuration for the multikey runner service.

### Acceptance Criteria
- [x] Modify `scripts/setup_gaul.sh` to check for and source a `.env` file if present
- [x] Add a step that prompts user or documents how to create `.env` with `GROQ_API_KEYS`
- [x] Ensure the `.env` file is in `.gitignore` (security)
- [x] Test that `GroqKeyPool` initializes correctly on GAUL with the configured keys

### Technical Notes
- Environment variable format: `GROQ_API_KEYS='["key1","key2","key3"]'` (JSON array)
- Optional: `GROQ_API_KEY_CONCURRENCY='{"key1":2,"key2":2}'` for per-key limits
- Reference implementation: `bailiff/agents/groq_pool.py:231-250`

### Files to Modify
- `scripts/setup_gaul.sh`
- `.gitignore` (verify `.env` is excluded)

---

## Ticket 2: Enable Groq Backend in GAUL Batch Configuration

**Priority:** High
**Labels:** `configuration`, `gaul`, `groq`
**Estimate:** 1 point

### Description
Update the GAUL batch configuration to use Groq backend instead of LOCAL, with appropriate rate limiting settings.

### Acceptance Criteria
- [ ] Uncomment and configure the Groq backend section in `configs/gaul_batch.yaml`
- [ ] Set `backend: groq` and `model: llama3-8b-8192`
- [ ] Configure conservative concurrency (start with 1)
- [ ] Add `backend_policy` with appropriate `rate_limit_seconds`
- [ ] Test configuration loads without errors

### Technical Notes
Example configuration:
```yaml
models:
  - backend: groq
    model: llama3-8b-8192
    backend_policy:
      max_retries: 5
      backoff_seconds: 2.0
      rate_limit_seconds: 1.0
concurrency: 1  # Start conservative, increase after testing
```

### Files to Modify
- `configs/gaul_batch.yaml`

---

## Ticket 3: Add Groq Key Pool Health Logging to Batch Runner

**Priority:** Medium
**Labels:** `observability`, `groq`, `logging`
**Estimate:** 2 points

### Description
Add periodic logging of Groq key pool status during batch experiment runs to help debug rate limiting issues.

### Acceptance Criteria
- [x] Import and access `GroqKeyPool` in the batch runner
- [x] Log pool summary at experiment start
- [x] Log pool summary periodically during runs (e.g., every 50 trials or 5 minutes)
- [x] Log pool summary at experiment end with totals
- [x] Include: keys in use, rate limit events, backoff states

### Technical Notes
- `GroqKeyPool` has a `summary()` method that returns key status snapshots
- Keys are redacted (shows last 6 chars only) for security
- Reference: `bailiff/agents/groq_pool.py:180-200`

### Files to Modify
- `scripts/run_trial_matrix.py`

---

## Ticket 4: Document Groq Backend Setup for GAUL

**Priority:** Medium
**Labels:** `documentation`, `gaul`, `groq`
**Estimate:** 1 point

### Description
Add documentation for setting up and using the Groq multikey runner on GAUL.

### Acceptance Criteria
- [x] Add "Optional: Groq Backend" section to `docs/GAUL_SETUP.md`
- [x] Document how to obtain and format Groq API keys
- [x] Document environment variable configuration
- [x] Include troubleshooting for common rate limit issues
- [x] Explain when to use Groq vs LOCAL backend

### Content to Include
1. How to get Groq API keys (free tier)
2. How to format `GROQ_API_KEYS` environment variable
3. How to securely transfer `.env` to GAUL
4. Recommended concurrency settings for free tier
5. How to monitor key pool health

### Files to Modify
- `docs/GAUL_SETUP.md`

---

## Ticket 5: Create Secure Key Deployment Script for GAUL

**Priority:** Low
**Labels:** `infrastructure`, `security`, `gaul`
**Estimate:** 2 points

### Description
Create a helper script to securely deploy Groq API keys to GAUL without committing them to git.

### Acceptance Criteria
- [x] Create `scripts/deploy_env_to_gaul.sh` (and `.ps1` for Windows)
- [x] Script should SCP `.env.local` to GAUL as `.env`
- [x] Validate `.env` format before deploying
- [x] Set correct file permissions (600) on remote
- [x] Print confirmation of successful deployment

### Technical Notes
- Use same SSH config as `deploy_to_gaul.sh`
- Target path: `~/ai-law-agents/.env`
- Never log or echo the actual key values

### Files to Create
- `scripts/deploy_env_to_gaul.sh`
- `scripts/deploy_env_to_gaul.ps1`

---

## Ticket 6: Add End-of-Experiment Groq Usage Report

**Priority:** Low
**Labels:** `observability`, `groq`, `reporting`
**Estimate:** 2 points

### Description
Generate a summary report at the end of batch experiments showing Groq API usage statistics.

### Acceptance Criteria
- [ ] Track total API calls per key during experiment
- [ ] Track total rate limit events per key
- [ ] Track total tokens consumed (if available from Groq response)
- [ ] Output summary to console and save to `results/groq_usage_report.json`
- [ ] Include experiment duration and average latency per call

### Technical Notes
- `GroqKeyStatus` tracks `total_uses` and `consecutive_rate_limits`
- Groq API responses include token usage in response metadata
- Reference: `bailiff/agents/groq_pool.py:30-60`

### Files to Modify
- `scripts/run_trial_matrix.py`
- `bailiff/agents/groq_pool.py` (may need to expose additional metrics)

---

## Summary Table

| # | Ticket | Priority | Estimate | Dependencies |
|---|--------|----------|----------|--------------|
| 1 | Configure Groq API Keys on GAUL | High | 1 pt | None |
| 2 | Enable Groq Backend in Config | High | 1 pt | Ticket 1 |
| 3 | Add Key Pool Health Logging | Medium | 2 pts | Ticket 2 |
| 4 | Document Groq Setup for GAUL | Medium | 1 pt | Tickets 1-2 |
| 5 | Secure Key Deployment Script | Low | 2 pts | Ticket 1 |
| 6 | End-of-Experiment Usage Report | Low | 2 pts | Ticket 3 |

**Total: 9 points**

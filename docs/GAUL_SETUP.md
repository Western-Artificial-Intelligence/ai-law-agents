# Running Experiments on GAUL

This guide details how to deploy and run the bias analysis experiments on the Western GAUL compute cluster.

## Prerequisites
- **GAUL Account**: You must have access to `compute.gaul.csd.uwo.ca`.
- **SSH Client**: PuTTY (Windows) or Terminal (Mac/Linux).
- **VPN**: If off-campus, ensure you are connected to Western ROAMS (Pulse Secure).

## 1. Deployment

### Option A: Automated Deployment (Recommended)
Run the deployment script from your local machine (Git Bash or Terminal):

```bash
# Replace 'your_username' with your Western ID
./scripts/deploy_to_gaul.sh your_username
```

### Option B: Manual Upload
If the script fails, use WinSCP or FileZilla to copy the `ai-law-agents` folder to your home directory on GAUL.

### Optional: Deploy Groq API Keys
If you keep Groq keys in a local `.env.local`, you can deploy it securely:

```bash
./scripts/deploy_env_to_gaul.sh your_username
```

This uploads `.env.local` to GAUL as `~/ai-law-agents/.env` and sets permissions to `600`.

## 2. Setup on GAUL

1. **SSH into GAUL GPU Node**:
   ```bash
   ssh your_username@gpu1.gaul.csd.uwo.ca
   ```

2. **Navigate to the directory**:
   ```bash
   cd ai-law-agents
   ```

3. **(Optional) Configure Groq API keys**:
   If you plan to run Groq-backed batches, create a local `.env` on GAUL before setup:
   ```bash
   cat <<'EOF' > .env
   GROQ_API_KEYS='["key1","key2","key3"]'
   # Optional per-key limits
   GROQ_API_KEY_CONCURRENCY='{"key1":2,"key2":2}'
   EOF
   ```
   The setup script will source `.env` if present and validate the key pool.

4. **Run the setup script**:
   ```bash
   chmod +x scripts/setup_gaul.sh
   ./scripts/setup_gaul.sh
   ```

5. **Authenticate with Hugging Face**:
   *Required for Llama-3 models.*
   
   **Prerequisite**: You must accept the license agreement at [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
   
   ```bash
   source .venv/bin/activate
   huggingface-cli login
   # Paste your HF token when prompted (get it from https://huggingface.co/settings/tokens)
   ```

## 3. Running Experiments

We use `nohup` to run experiments in the background so they continue even if your SSH connection drops.

1. **Start the batch**:
   ```bash
   chmod +x scripts/run_gaul_experiments.sh
   ./scripts/run_gaul_experiments.sh
   ```

2. **Monitor Progress**:
   The script will output a log file path (e.g., `runs/execution_20231201_120000.log`).
   ```bash
   tail -f runs/execution_YYYYMMDD_HHMMSS.log
   ```

3. **Check GPU Usage**:
   ```bash
   nvidia-smi
   ```

## 4. Retrieving Results

After the experiments complete (check the log file), download the results to your local machine:

```bash
# Run locally
scp your_username@compute.gaul.csd.uwo.ca:~/ai-law-agents/runs/*.jsonl ./runs/
```

## Configuration Details
The experiment is configured in `configs/gaul_batch.yaml`:
- **Cases**: 6 cases (Traffic, Assault, Shoplifting, DUI, Vandalism, Petty Theft)
- **Cues**: Black, Chinese, Indian names vs. White control
- **Seeds**: 30 replications per case/cue pair
- **Backend**: Local Llama-3-8B (running on GAUL GPU)

## Optional: Groq Backend
Use Groq when you want faster iteration or to avoid tying up GAUL GPUs. Use LOCAL when you need GPU-local models, deterministic offline runs, or to avoid external API limits.

### 1. Get Groq API keys
1. Create a free Groq account and generate API keys.
2. Keep keys out of git by storing them in a local `.env.local`.

### 2. Format keys for multi-key runner
In `.env.local`, set a JSON list of keys:
```bash
GROQ_API_KEYS='["key1","key2","key3"]'
```
Optional per-key limits (recommended for free tier):
```bash
GROQ_API_KEY_CONCURRENCY='{"key1":1,"key2":1}'
```
Suggested starting point: 1 concurrent request per key. Increase only if Groq dashboard limits allow it.

### 3. Securely transfer keys to GAUL
Use the helper script (recommended):
```bash
./scripts/deploy_env_to_gaul.sh your_username
```
This uploads `.env.local` as `~/ai-law-agents/.env` and sets permissions to `600`.

### 4. Run Groq-backed batches
Update `configs/gaul_batch.yaml` to use `backend: groq` and a Groq model (e.g., `llama3-8b-8192`), then run:
```bash
./scripts/run_gaul_experiments.sh
```

### 5. Monitor key pool health
The batch runner logs Groq key pool status at start, periodically, and at the end. Look for `[GROQ]` entries in the execution log for:
- inflight requests
- rate limit streaks
- backoff remaining
- total uses

### Troubleshooting rate limits
- If you see repeated backoffs or long delays, reduce concurrency to 1 per key.
- Verify `GROQ_API_KEYS` is valid JSON (no trailing commas).
- Ensure key limits match your Groq dashboard quotas.

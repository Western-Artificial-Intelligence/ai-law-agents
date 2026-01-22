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

## 2. Setup on GAUL

1. **SSH into GAUL GPU Node**:
   ```bash
   ssh your_username@gpu1.gaul.csd.uwo.ca
   ```

2. **Navigate to the directory**:
   ```bash
   cd ai-law-agents
   ```

3. **Run the setup script**:
   ```bash
   chmod +x scripts/setup_gaul.sh
   ./scripts/setup_gaul.sh
   ```

4. **Authenticate with Hugging Face**:
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

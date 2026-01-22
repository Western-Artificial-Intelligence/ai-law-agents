#!/bin/bash
# Script to deploy .env.local to GAUL as .env
# Usage: ./scripts/deploy_env_to_gaul.sh <username>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <gaul_username>"
    exit 1
fi

USER=$1
HOST="gpu1.gaul.csd.uwo.ca"
REMOTE_DIR="~/ai-law-agents"
LOCAL_ENV=".env.local"
REMOTE_ENV="$REMOTE_DIR/.env"

if [ ! -f "$LOCAL_ENV" ]; then
    echo "ERROR: $LOCAL_ENV not found."
    exit 1
fi

python3 - "$LOCAL_ENV" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_text(encoding="utf-8")
env = {}
for line in data.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    env[key] = value

if "GROQ_API_KEYS" in env:
    try:
        parsed = json.loads(env["GROQ_API_KEYS"])
    except json.JSONDecodeError as exc:
        raise SystemExit("ERROR: GROQ_API_KEYS must be valid JSON.") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) and item.strip() for item in parsed):
        raise SystemExit("ERROR: GROQ_API_KEYS must be a JSON list of non-empty strings.")
elif "GROQ_API_KEY" in env:
    if not env["GROQ_API_KEY"].strip():
        raise SystemExit("ERROR: GROQ_API_KEY cannot be empty.")
else:
    raise SystemExit("ERROR: Missing GROQ_API_KEYS or GROQ_API_KEY in .env.local.")

if "GROQ_API_KEY_CONCURRENCY" in env:
    try:
        parsed = json.loads(env["GROQ_API_KEY_CONCURRENCY"])
    except json.JSONDecodeError as exc:
        raise SystemExit("ERROR: GROQ_API_KEY_CONCURRENCY must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("ERROR: GROQ_API_KEY_CONCURRENCY must be a JSON object.")
    for value in parsed.values():
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit("ERROR: GROQ_API_KEY_CONCURRENCY values must be integers.") from exc
        if limit <= 0:
            raise SystemExit("ERROR: GROQ_API_KEY_CONCURRENCY values must be positive integers.")

print("OK: .env.local validation passed.")
PY

echo "Deploying .env.local to $USER@$HOST..."
ssh "$USER@$HOST" "mkdir -p $REMOTE_DIR"
scp -p "$LOCAL_ENV" "${USER}@${HOST}:${REMOTE_ENV}"
ssh "$USER@$HOST" "chmod 600 $REMOTE_ENV"

echo "Deployment complete. Remote .env stored at $REMOTE_ENV with permissions 600."

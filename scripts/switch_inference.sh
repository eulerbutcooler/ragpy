#!/usr/bin/env bash
# Switch RAGPy inference between the Modal test stack and local servers.
# Usage: ./scripts/switch_inference.sh modal|local|sglang
set -euo pipefail

profile="${1:-}"
case "$profile" in
  modal|local|sglang) ;;
  *)
    echo "Usage: $0 modal|local|sglang" >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")/.."

if [[ ! -f "env/inference.${profile}.env" ]]; then
  echo "Missing profile: env/inference.${profile}.env" >&2
  exit 1
fi

python3 - "$profile" <<'PY'
import sys
from pathlib import Path

profile = sys.argv[1]
updates = {}
for line in Path(f"env/inference.{profile}.env").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    updates[key.strip()] = value.strip()

env_path = Path(".env")
lines = env_path.read_text().splitlines()
out = []
seen = set()
for line in lines:
    if "=" in line and not line.lstrip().startswith("#"):
        key = line.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
            continue
    out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

env_path.write_text("\n".join(out) + "\n")
print(f"Active inference profile: {profile}")
print("Updated: " + ", ".join(sorted(updates)))
PY

echo "Restart RAGPy to apply: stop the uvicorn process and run it again."

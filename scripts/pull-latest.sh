#!/usr/bin/env bash
# Pull the latest deployment branch for all three repositories.
#
# Usage:
#   ./ragpy/scripts/pull-latest.sh
#   ROOT=/srv/aeromentor BRANCH=main ./ragpy/scripts/pull-latest.sh
#
# The upstream URLs are the canonical eulerbutcooler repositories. If the
# client cloned those directly, origin already points there; pulling from
# upstream is still safe.

set -euo pipefail

ROOT="${ROOT:-$HOME/aeromentor}"
BRANCH="${BRANCH:-dev}"

[[ -d "$ROOT" ]] || {
  echo "ERROR: $ROOT not found."
  exit 1
}

BACKEND_DIR="$ROOT/wingman-backend"
if [[ ! -d "$BACKEND_DIR" && -d "$ROOT/backend" ]]; then
  BACKEND_DIR="$ROOT/backend"
fi

pull_repo() {
  local dir="$1" url="$2" label="$3"

  [[ -d "$dir/.git" ]] || {
    echo "ERROR: $dir is not a git repository."
    exit 1
  }

  echo
  echo "== $label =="
  cd "$dir"

  if git remote get-url upstream >/dev/null 2>&1; then
    git remote set-url upstream "$url"
  else
    git remote add upstream "$url"
  fi

  git fetch upstream "$BRANCH"

  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH" --track "upstream/$BRANCH"
  fi

  git pull --ff-only upstream "$BRANCH"
}

pull_repo "$ROOT/frontend" \
  "https://github.com/eulerbutcooler/frontend.git" "frontend"

pull_repo "$BACKEND_DIR" \
  "https://github.com/eulerbutcooler/backend.git" "backend"

pull_repo "$ROOT/ragpy" \
  "https://github.com/eulerbutcooler/ragpy.git" "ragpy"

echo
echo "Code updated. Complete the deployment with:"
echo "  cd $ROOT/frontend && npm ci && npm run build"
echo "  cd $ROOT/ragpy && uv sync"
echo "  cd $ROOT && ./ragpy/scripts/dev.sh"

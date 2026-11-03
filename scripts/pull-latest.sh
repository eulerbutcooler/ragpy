#!/usr/bin/env bash
# Pull the latest deployment branch for all three repositories.
#
# Usage:
#   ./ragpy/scripts/pull-latest.sh
#   REMOTE=upstream ./ragpy/scripts/pull-latest.sh
#   ROOT=/srv/aeromentor BRANCH=main ./ragpy/scripts/pull-latest.sh
#
# Each repository's configured origin is used by default. Set REMOTE=upstream
# if you maintain a fork and want to pull from the canonical repository.

set -euo pipefail

ROOT="${ROOT:-$HOME/aeromentor}"
BRANCH="${BRANCH:-dev}"
REMOTE="${REMOTE:-origin}"

[[ -d "$ROOT" ]] || {
  echo "ERROR: $ROOT not found."
  exit 1
}

BACKEND_DIR="$ROOT/wingman-backend"
if [[ ! -d "$BACKEND_DIR" && -d "$ROOT/backend" ]]; then
  BACKEND_DIR="$ROOT/backend"
fi

pull_repo() {
  local dir="$1" label="$2"

  [[ -d "$dir/.git" ]] || {
    echo "ERROR: $dir is not a git repository."
    exit 1
  }

  echo
  echo "== $label =="
  cd "$dir"

  git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "ERROR: remote '$REMOTE' not found in $dir."
    echo "Set REMOTE=origin or add the remote first."
    exit 1
  }

  git fetch "$REMOTE"

  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git checkout "$BRANCH"
  else
    git checkout -b "$BRANCH" --track "$REMOTE/$BRANCH"
  fi

  git pull --ff-only "$REMOTE" "$BRANCH"
}

pull_repo "$ROOT/frontend" "frontend"
pull_repo "$BACKEND_DIR" "backend"
pull_repo "$ROOT/ragpy" "ragpy"

echo
echo "Code updated. Complete the deployment with:"
echo "  cd $ROOT && INSTALL=1 ./ragpy/scripts/dev.sh"

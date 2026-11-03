#!/usr/bin/env bash
# Build llama.cpp with CUDA, download Qwen3.5-9B-IQ4_XS.gguf, and start
# llama-server on :8081.
#
# Usage:
#   ./ragpy/scripts/setup-llama.sh
#   ./ragpy/scripts/setup-llama.sh --no-start
#   ./ragpy/scripts/setup-llama.sh --stop
#
# Options:
#   --no-start        Install/build/download only; do not start llama-server
#   --stop            Stop the setup-managed tmux session "llama"
#   --rebuild         Delete the llama.cpp build directory and rebuild
#   --force-download  Re-download the GGUF even if it exists
#   --skip-apt        Do not install missing build tools with apt
#   --no-cuda-patch   Do not patch CUDA headers for newer glibc
#   --port PORT       llama-server port (default: 8081)
#   --model-dir DIR   Directory for the GGUF (default: ~/models)
#   --llama-dir DIR   llama.cpp checkout (default: ~/llama.cpp)
#
# This script requires the NVIDIA driver and CUDA toolkit to be installed
# already (nvidia-smi and nvcc must work). It does not install the driver.

set -euo pipefail

SESSION="${SESSION:-llama}"
PORT="${PORT:-8081}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-9B-GGUF}"
MODEL_REVISION="${MODEL_REVISION:-3885219b6810b007914f3a7950a8d1b469d598a5}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-9B-IQ4_XS.gguf}"
JOBS="${JOBS:-$(nproc)}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

START=1
STOP=0
REBUILD=0
FORCE_DOWNLOAD=0
INSTALL_DEPS=1
PATCH_CUDA=1
CUDA_HEADER_PATCHED=0
SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"

usage() {
  sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# CUDA 12.x headers conflict with glibc >= 2.38/2.41, which declares cospi,
# sinpi and rsqrt with noexcept(true). This is the fix recommended by
# llama.cpp's build docs. A backup of the header is kept.
patch_cuda_math_header() {
  [[ "$PATCH_CUDA" == "1" ]] || return 0

  local nvcc_real cuda_home header tmp backup count candidate

  nvcc_real="$(readlink -f "$(command -v nvcc)")"
  cuda_home="$(cd "$(dirname "$nvcc_real")/.." && pwd)"

  header=""
  for candidate in \
    "$cuda_home/targets/x86_64-linux/include/crt/math_functions.h" \
    "$cuda_home/include/crt/math_functions.h" \
    "/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h" \
    "/usr/local/cuda/include/crt/math_functions.h"; do
    if [[ -f "$candidate" ]]; then
      header="$candidate"
      break
    fi
  done
  [[ -n "$header" ]] || die "CUDA math_functions.h not found; cannot apply glibc compatibility patch."

  tmp="$(mktemp)"
  cp "$header" "$tmp"

  count="$(python3 - "$tmp" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()
names = r"cospi|cospif|sinpi|sinpif|rsqrt|rsqrtf"

def add_noexcept(pattern, text):
    changed = 0
    def repl(match):
        nonlocal changed
        decl = match.group(1)
        if "noexcept" in decl or "__NV_IEC_60559_FUNCS_EXCEPTION_SPECIFIER" in decl:
            return match.group(0)
        changed += 1
        return decl + " noexcept (true)" + match.group(2)
    return pattern.sub(repl, text), changed

first = re.compile(
    r"(extern\s+__DEVICE_FUNCTIONS_DECL__\s+__device_builtin__\s+"
    r"(?:double|float)\s+(?:" + names + r")\s*\([^;]*?\))(\s*;)"
)
text, count = add_noexcept(first, text)

fallback = re.compile(
    r"(extern\s+[^;\n]*?\b(?:" + names + r")\s*\([^;]*?\))(\s*;)"
)
text, more = add_noexcept(fallback, text)
count += more

open(path, "w").write(text)
print(count)
PY
)" || die "failed to apply CUDA header compatibility patch."

  if [[ "$count" == "0" ]]; then
    echo "CUDA math header already compatible; no patch needed."
    rm -f "$tmp"
    return 0
  fi

  backup="${header}.aeromentor.bak"
  if [[ ! -f "$backup" ]]; then
    $SUDO cp -p "$header" "$backup" || die "could not back up $header"
  fi
  $SUDO install -m 0644 "$tmp" "$header" || die "could not write $header"
  rm -f "$tmp"

  CUDA_HEADER_PATCHED=1
  echo "Patched $count CUDA math declaration(s) for glibc compatibility."
  echo "Backup: $backup"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-start) START=0; shift ;;
    --stop) STOP=1; shift ;;
    --rebuild) REBUILD=1; shift ;;
    --force-download) FORCE_DOWNLOAD=1; shift ;;
    --skip-apt) INSTALL_DEPS=0; shift ;;
    --no-cuda-patch) PATCH_CUDA=0; shift ;;
    --port) PORT="${2:?missing value}"; shift 2 ;;
    --model-dir) MODEL_DIR="${2:?missing value}"; shift 2 ;;
    --llama-dir) LLAMA_DIR="${2:?missing value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

GGUF_PATH="$MODEL_DIR/$MODEL_FILE"

if [[ "$STOP" == "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session '$SESSION'."
  else
    echo "tmux session '$SESSION' is not running."
  fi
  exit 0
fi

# --- Prerequisites ---------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || \
  die "nvidia-smi not found. Install the NVIDIA driver first."
nvidia-smi >/dev/null 2>&1 || \
  die "nvidia-smi failed. Fix the NVIDIA driver before continuing."

command -v nvcc >/dev/null 2>&1 || \
  die "nvcc not found. Install the CUDA toolkit matching your NVIDIA driver first."
nvcc --version >/dev/null 2>&1 || \
  die "nvcc failed. Fix the CUDA toolkit before continuing."

missing=()
for cmd in git cmake make g++ curl tmux python3; do
  command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done

if (( ${#missing[@]} > 0 )); then
  if [[ "$INSTALL_DEPS" != "1" ]]; then
    die "missing build tools: ${missing[*]} (rerun without --skip-apt)"
  fi
  command -v apt-get >/dev/null 2>&1 || \
    die "missing build tools: ${missing[*]} (install them manually)"
  echo "Installing build tools: ${missing[*]}"
  $SUDO apt-get update
  $SUDO apt-get install -y \
    build-essential cmake git curl libcurl4-openssl-dev pkg-config tmux python3
fi

patch_cuda_math_header
if [[ "$CUDA_HEADER_PATCHED" == "1" ]]; then
  REBUILD=1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed."
command -v tmux >/dev/null 2>&1 || die "tmux is required."

# --- llama.cpp -------------------------------------------------------------
if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  [[ -e "$LLAMA_DIR" ]] && \
    die "$LLAMA_DIR exists but is not a git checkout."
  echo "Cloning llama.cpp into $LLAMA_DIR..."
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  echo "Updating llama.cpp..."
  git -C "$LLAMA_DIR" fetch --all --prune
  git -C "$LLAMA_DIR" pull --ff-only
fi

if [[ "$REBUILD" == "1" ]]; then
  echo "Removing old build directory..."
  rm -rf "$LLAMA_DIR/build"
fi

echo "Configuring llama.cpp with CUDA..."
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

echo "Building llama.cpp with $JOBS jobs (this can take a while)..."
cmake --build "$LLAMA_DIR/build" --config Release -j"$JOBS"

LLAMA_SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
[[ -x "$LLAMA_SERVER_BIN" ]] || die "llama-server was not built at $LLAMA_SERVER_BIN"

mkdir -p "$BIN_DIR"
ln -sf "$LLAMA_SERVER_BIN" "$BIN_DIR/llama-server"
export PATH="$BIN_DIR:$PATH"
echo "llama-server: $LLAMA_SERVER_BIN"
echo "Symlinked:   $BIN_DIR/llama-server"

# --- Model -----------------------------------------------------------------
mkdir -p "$MODEL_DIR"

if [[ "$FORCE_DOWNLOAD" == "1" || ! -s "$GGUF_PATH" ]]; then
  echo "Downloading $MODEL_FILE (~5.2 GB) into $MODEL_DIR..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" \
      --revision "$MODEL_REVISION" \
      --local-dir "$MODEL_DIR"
  elif uvx --from "huggingface_hub[cli]" huggingface-cli --help >/dev/null 2>&1; then
    uvx --from "huggingface_hub[cli]" huggingface-cli download \
      "$MODEL_REPO" "$MODEL_FILE" \
      --revision "$MODEL_REVISION" \
      --local-dir "$MODEL_DIR"
  else
    uvx --from "huggingface_hub[cli]" hf download \
      "$MODEL_REPO" "$MODEL_FILE" \
      --revision "$MODEL_REVISION" \
      --local-dir "$MODEL_DIR"
  fi
else
  echo "Using existing model: $GGUF_PATH"
fi

[[ -s "$GGUF_PATH" ]] || die "model not found at $GGUF_PATH"
SIZE="$(stat -c%s "$GGUF_PATH")"
(( SIZE > 4 * 1024 * 1024 * 1024 )) || \
  die "model looks incomplete ($SIZE bytes) at $GGUF_PATH"
echo "Model: $GGUF_PATH ($(numfmt --to=iec-i --suffix=B "$SIZE"))"

if [[ "$START" != "1" ]]; then
  echo
  echo "Setup complete. Start the app with:"
  echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  echo "  ./ragpy/scripts/dev.sh"
  exit 0
fi

# --- Start -----------------------------------------------------------------
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "llama-server is already healthy on :$PORT; leaving it running."
  exit 0
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

llama_cmd=(
  "$LLAMA_SERVER_BIN"
  -m "$GGUF_PATH"
  --host 127.0.0.1
  --port "$PORT"
  --ctx-size 98304
  --parallel 3
  --n-gpu-layers 99
  --flash-attn on
  --cache-type-k q8_0
  --cache-type-v q8_0
  --batch-size 1024
  --ubatch-size 256
  --jinja
  --reasoning off
  --alias Qwen3.5-9B-IQ4_XS.gguf
)
printf -v llama_cmd_str '%q ' "${llama_cmd[@]}"

echo "Starting llama-server on :$PORT..."
tmux new-session -d -s "$SESSION" -c "$HOME" "$llama_cmd_str; exec bash"

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo
    echo "llama-server is running on http://127.0.0.1:$PORT"
    echo "Logs:  tmux attach -t $SESSION"
    echo "Stop:  $(basename "$0") --stop"
    echo
    echo "Start the app with:"
    echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo "  ./ragpy/scripts/dev.sh"
    exit 0
  fi
  sleep 2
done

echo "llama-server did not become healthy in time. Last logs:" >&2
tmux capture-pane -p -t "$SESSION" 2>/dev/null | tail -60 >&2 || true
exit 1

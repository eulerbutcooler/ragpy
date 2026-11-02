# Server setup (client deployment)

This is the deployment runbook for the GPU server that runs AeroMentor.
The server layout is expected to be:

```text
~/aeromentor/
├── wingman-backend/   # Go API
├── ragpy/             # RAG + inference service
└── frontend/          # Next.js app
```

The one-command runner is `~/aeromentor/ragpy/scripts/dev.sh`.

---

## 1. Server prerequisites

- Ubuntu 22.04 / 24.04
- NVIDIA GPU with a working driver (`nvidia-smi`)
- CUDA toolchain for `llama.cpp`
- Docker Engine + Compose plugin
- Go **1.25+**
- Node.js **20.9+**
- Python **3.11+** (3.12 recommended) and [uv](https://docs.astral.sh/uv/)
- `git`, `curl`, `tmux`
- `build-essential`, `cmake` (for building llama.cpp)
- `llama-server` built with CUDA, and the GGUF model:
  `/models/Qwen3.5-9B-IQ4_XS.gguf` (~5.2 GB)

For SGLang mode only: NVIDIA Container Toolkit and the `lmsysorg/sglang` image.

### What the runner does and does not install

`ragpy/scripts/dev.sh` is a **runner**, not an installer. On each start it:

- runs `docker compose up -d` (this pulls missing container images)
- runs `make inference-local` / `inference-sglang` / `inference-modal`
- starts llama.cpp or SGLang, the backend, RAGPy, and the frontend

It does **not** install:

- the NVIDIA driver or CUDA
- `llama-server` or the GGUF model
- Go, Node.js, Python, `uv`, Docker, or tmux
- Python, Go, or npm dependencies
- the production frontend build

Provision those once with the steps below.

### One-command llama.cpp setup

From `~/aeromentor`:

```bash
./ragpy/scripts/setup-llama.sh
```

This script:

- installs missing build tools if needed (`--skip-apt` to disable)
- clones/builds `llama.cpp` with `-DGGML_CUDA=ON`
- installs `llama-server` into `~/.local/bin`
- downloads `Qwen3.5-9B-IQ4_XS.gguf` (~5.2 GB) into `~/models`
- starts `llama-server` on `:8081` in tmux session `llama`
- waits until `/health` is ready

Then run:

```bash
./ragpy/scripts/dev.sh
```

`dev.sh` detects the running server and reuses it instead of starting a
second one.

Useful options:

```bash
./ragpy/scripts/setup-llama.sh --no-start        # build/download only
./ragpy/scripts/setup-llama.sh --stop            # stop tmux session "llama"
./ragpy/scripts/setup-llama.sh --rebuild         # clean rebuild
./ragpy/scripts/setup-llama.sh --force-download  # re-download the GGUF
./ragpy/scripts/setup-llama.sh --model-dir /models
```

It requires the NVIDIA driver and CUDA toolkit (`nvidia-smi` and `nvcc`)
to be installed already.

### Manual alternative

If you prefer to build it yourself:

```bash
# Check the GPU stack first.
nvidia-smi
nvcc --version

# Build llama.cpp with CUDA.
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build --config Release -j"$(nproc)"
sudo install -m 0755 ~/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Download the GGUF used by Modal and the local profile.
mkdir -p /models
uvx --from "huggingface_hub[cli]" huggingface-cli download \
  unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-IQ4_XS.gguf \
  --revision 3885219b6810b007914f3a7950a8d1b469d598a5 \
  --local-dir /models
```

If `/models` is not writable, download to `~/models` and set
`GGUF_PATH=~/models/Qwen3.5-9B-IQ4_XS.gguf` when running `dev.sh`.

If `nvidia-smi` or `nvcc` is missing, install the NVIDIA driver and CUDA
toolkit first. The runner cannot install those safely.

---

## 2. Clone the repositories

```bash
mkdir -p ~/aeromentor
cd ~/aeromentor

git clone https://github.com/eulerbutcooler/backend.git  wingman-backend
git clone https://github.com/eulerbutcooler/frontend.git frontend
git clone https://github.com/eulerbutcooler/ragpy.git    ragpy
```

The runner also accepts `backend/` if you clone it under that name.

---

## 3. Configure environment files

### Backend: `~/aeromentor/wingman-backend/.env`

```bash
cd ~/aeromentor/wingman-backend
cp .env.example .env
```

Set at minimum:

```dotenv
DATABASE_URL=postgres://aeromentor:aeromentor@localhost:5432/aeromentor?sslmode=disable
REDIS_ADDR=localhost:6379
NATS_URL=nats://localhost:4222
MINIO_ENDPOINT=localhost:9000
RAG_BASE_URL=http://localhost:8000

# Must match INTERNAL_TOKEN in ragpy/.env
RAG_INTERNAL_TOKEN=<shared-internal-token>

JWT_ACCESS_SECRET=<random-long-secret>
JWT_REFRESH_SECRET=<random-long-secret>

# Browser origin that calls the backend directly for uploads.
# Example: http://203.0.113.10:3000 or https://app.example.com
CORS_ALLOWED_ORIGINS=http://<SERVER_IP>:3000

LOG_LEVEL=info
```

Keep the MinIO access/secret values from `.env.example` unless you changed the
storage service. The Docker Compose stack creates Postgres/Redis/NATS/Qdrant/
SeaweedFS automatically.

### RAG service: `~/aeromentor/ragpy/.env`

```bash
cd ~/aeromentor/ragpy
cp .env.example .env
```

Set/make sure these match the backend:

```dotenv
INTERNAL_TOKEN=<same value as RAG_INTERNAL_TOKEN>
QDRANT_URL=http://localhost:6333
NATS_URL=nats://localhost:4222
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY_ID=<same as backend MINIO_ACCESS_KEY>
MINIO_SECRET_ACCESS_KEY=<same as backend MINIO_SECRET_KEY>
```

The LLM profile is selected by the runner:

```bash
make inference-local   # llama.cpp on :8081; default
make inference-sglang  # SGLang on :1234
make inference-modal   # Modal endpoint
```

`make inference-local` writes the LLM/embedding/reranker keys into `.env`.
The runner does this automatically before starting RAGPy.

For Modal, create the ignored profile from the example and fill in the URL/key:

```bash
cp env/inference.modal.env.example env/inference.modal.env
# edit LLM_BASE_URL and LLM_API_KEY
```

### Frontend: `~/aeromentor/frontend/.env`

```dotenv
NEXT_PUBLIC_API_URL=
NEXT_PUBLIC_UPLOAD_URL=http://<SERVER_IP>:8080
API_URL=http://localhost:8080
NEXT_PUBLIC_MINIO_HOST=localhost
NEXT_PUBLIC_MINIO_PORT=9000
AUTH_SECRET=<openssl rand -base64 32>
AUTH_TRUST_HOST=true
```

Notes:

- `NEXT_PUBLIC_API_URL` must stay **empty**. Browser API calls go to the
  same origin and Next proxies `/api/v1` to the backend.
- `NEXT_PUBLIC_UPLOAD_URL` is used for TUS file uploads. It must be reachable
  from the **browser**, not just the server. Use the server IP or HTTPS API
  domain.
- `NEXT_PUBLIC_*` values are baked in at build time. Rebuild the frontend
  after changing them.
- Do **not** set `AUTH_URL`; the app derives the origin from the request and
  `AUTH_TRUST_HOST=true` handles the proxy.

---

## 4. Install and build

```bash
# Backend
cd ~/aeromentor/wingman-backend
go mod download

# RAG service
cd ~/aeromentor/ragpy
uv sync
# Optional for Modal deployment:
# uv sync --group deploy

# Frontend
cd ~/aeromentor/frontend
npm ci
npm run build
```

---

## 5. Run

```bash
cd ~/aeromentor
./ragpy/scripts/dev.sh
```

The default is local llama.cpp. It starts:

- llama.cpp on `:8081`
- Go backend on `:8080`
- RAGPy on `:8000`
- Next.js on `:3000`
- Postgres/Redis/NATS/Qdrant/SeaweedFS through Docker

The processes live in the `aeromentor` tmux session:

```bash
tmux attach -t aeromentor   # view logs
# Ctrl-b d to detach
```

Health checks:

```bash
curl localhost:8080/health
curl localhost:8000/api/v1/health
curl localhost:3000/login
```

Open `http://<SERVER_IP>:3000`.

### Inference modes

```bash
./ragpy/scripts/dev.sh                  # llama.cpp (default)
INFERENCE=sglang ./ragpy/scripts/dev.sh # SGLang + Qwen2.5-7B-AWQ
INFERENCE=modal ./ragpy/scripts/dev.sh  # Modal deploy
INFERENCE=none ./ragpy/scripts/dev.sh   # no AI (UI/API only)
```

Overrides:

```bash
GGUF_PATH=/data/models/Qwen3.5-9B-IQ4_XS.gguf \
LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server \
./ragpy/scripts/dev.sh
```

---

## 6. Public access / HTTPS

For a direct IP deployment, expose only:

- `3000/tcp` — web app
- `8080/tcp` — API / uploads

Keep Postgres (`5432`), Redis (`6379`), NATS (`4222`), Qdrant (`6333`) and
SeaweedFS (`9000`) private.

For HTTPS with Caddy:

```caddyfile
app.example.com {
    reverse_proxy 127.0.0.1:3000
}

api.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

Then set:

```dotenv
# frontend/.env
NEXT_PUBLIC_UPLOAD_URL=https://api.example.com

# wingman-backend/.env
CORS_ALLOWED_ORIGINS=https://app.example.com
```

Rebuild the frontend and restart the backend. The app's API calls remain
same-origin through Next; only uploads use `api.example.com`.

Do not mix HTTPS app + HTTP upload URL — browsers block mixed content.

---

## 7. Updating the server

```bash
cd ~/aeromentor
./ragpy/scripts/pull-latest.sh
```

The script updates all three repositories from the canonical `upstream`
repositories on branch `dev`:

| Directory | Upstream |
|---|---|
| `frontend` | `https://github.com/eulerbutcooler/frontend.git` |
| `wingman-backend` | `https://github.com/eulerbutcooler/backend.git` |
| `ragpy` | `https://github.com/eulerbutcooler/ragpy.git` |

Then reinstall/build and restart:

```bash
cd ~/aeromentor/frontend && npm ci && npm run build
cd ~/aeromentor/ragpy && uv sync
cd ~/aeromentor
./ragpy/scripts/dev.sh
```

The runner kills the old `aeromentor` tmux session and starts everything
again. If the branch is `main`:

```bash
BRANCH=main ./ragpy/scripts/pull-latest.sh
```

Manual equivalent, per repository:

```bash
cd ~/aeromentor/frontend
git remote add upstream https://github.com/eulerbutcooler/frontend.git
git fetch upstream
git checkout dev
git pull --ff-only upstream dev

cd ~/aeromentor/wingman-backend
git remote add upstream https://github.com/eulerbutcooler/backend.git
git fetch upstream
git checkout dev
git pull --ff-only upstream dev

cd ~/aeromentor/ragpy
git remote add upstream https://github.com/eulerbutcooler/ragpy.git
git fetch upstream
git checkout dev
git pull --ff-only upstream dev
```

If `upstream` already exists, use `git remote set-url upstream <url>`.

---

## 8. Modal

To stop paying for the warm GPU:

```bash
cd ~/aeromentor/ragpy
uv run modal app stop --yes aeromentor-inference
```

To resume:

```bash
cd ~/aeromentor/ragpy
# First time on this server:
cp env/inference.modal.env.example env/inference.modal.env
# edit LLM_BASE_URL and LLM_API_KEY in that file

uv run modal deploy modal_inference.py
uv run python scripts/check_inference.py
INFERENCE=modal ./scripts/dev.sh
```

Building/deploying from scratch is documented in `ragpy/MODAL_INFERENCE.md`.

---

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| `llama-server: command not found` | Set `LLAMA_SERVER=/path/to/llama-server` |
| `GGUF not found` | Set `GGUF_PATH`; the default is `/models/Qwen3.5-9B-IQ4_XS.gguf` |
| Chat/generation fails | Inference server not running; check `:8081` or Modal |
| Uploads fail | `CORS_ALLOWED_ORIGINS` must match the browser origin exactly; `NEXT_PUBLIC_UPLOAD_URL` must be browser-reachable; rebuild frontend |
| Login redirects to localhost | Remove `AUTH_URL`; keep `AUTH_TRUST_HOST=true` |
| Frontend env changes ignored | `NEXT_PUBLIC_*` is build-time; run `npm run build` |
| Port already in use | `ss -ltnp \| grep -E '3000\|8000\|8080'`; the runner frees these ports |
| Chat has no sources | Qdrant is empty; upload and ingest course files first |
| PDF/file previews fail | Check the backend and SeaweedFS are healthy; all file URLs are proxied server-side |

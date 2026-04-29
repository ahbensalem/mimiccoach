# Deployment runbook — MimicCoach

The hackathon stack is three services:

| Layer | Service | Cost |
|---|---|---|
| Vector DB | **Qdrant Cloud** | Free tier (1 GB, plenty of headroom for our ~125 MB) |
| Backend  | **Modal**         | ~$30 / month free credit |
| Frontend | **Vercel**        | Free tier |

End-to-end deploy is roughly five minutes once accounts are wired.

## 0. Source control

```bash
cd /home/ahbensalem/Qdrant-competition
gh repo create mimiccoach --public --source=. --remote=origin --push
```

(Or `--private` while iterating; flip to public for submission.)

## 1. Qdrant Cloud

1. Sign up at <https://cloud.qdrant.io>.
2. Create a free-tier cluster.
3. Copy the **endpoint URL** and an **API key**.

## 2. Modal

```bash
# One-time auth
pip install -U modal
modal token new

# Wire secrets to the app — pulled in via os.environ at runtime
modal secret create qdrant \
  QDRANT_URL=https://<cluster>.qdrant.io:6333 \
  QDRANT_API_KEY=<your-key> \
  FRONTEND_ORIGIN=https://<your-vercel-url>
```

Update `backend/app.py` to attach the secret to the long-lived functions:

```python
@app.function(timeout=300, max_containers=4, secrets=[modal.Secret.from_name("qdrant")])
@modal.asgi_app()
def fastapi_app():
    ...

@app.function(schedule=modal.Period(minutes=4), secrets=[modal.Secret.from_name("qdrant")])
def warm_keep() -> None:
    ...
```

Then build the reference library *into the Cloud cluster* and deploy:

```bash
QDRANT_URL=https://<cluster>.qdrant.io:6333 \
  QDRANT_API_KEY=<your-key> \
  ./scripts/build_library.sh

modal deploy backend/app.py
```

Modal prints the deployed URL — something like
`https://<workspace>--mimiccoach-fastapi-app.modal.run`. Save it.

After deploy, set the public URL the warm-keep job pings:

```bash
modal secret create mimiccoach-warm \
  MIMICCOACH_PUBLIC_URL=https://<workspace>--mimiccoach-fastapi-app.modal.run
```

(Or update the existing `qdrant` secret with that key.)

## 3. Vercel

```bash
cd frontend
vercel link
vercel env add MODAL_BACKEND_URL production
# paste the Modal URL when prompted
vercel --prod
```

## 4. Smoke test

```bash
# Backend health from the public URL
curl https://<workspace>--mimiccoach-fastapi-app.modal.run/healthz

# Motions list
curl https://<workspace>--mimiccoach-fastapi-app.modal.run/motions | jq .

# Frontend
open https://<your-vercel-url>
```

Upload a phone clip. The Vercel `/api/proxy/analyze` route will re-stream it
to Modal, which runs MediaPipe → segment → embed → Qdrant query → coach.
First call after a long idle takes ~10–30s for cold-start; subsequent calls
are sub-second on the analyze path itself plus pose extraction time.

## 5. Demo-day operations

- **Warm-keep**: the Modal `warm_keep` scheduled function pings `/healthz`
  every 4 minutes. No external cron needed.
- **Container scaling**: `max_containers=4` is set on `fastapi_app` —
  enough headroom for live judging, well under any free-tier limits.
- **Manifest reload**: if you need to refresh the reference library, run
  `./scripts/build_library.sh` again with `QDRANT_URL`/`QDRANT_API_KEY`
  set; this no-ops on schema (idempotent) and re-upserts points.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `MODAL_BACKEND_URL not configured` (frontend `/api/proxy`) | The Vercel env var isn't set or wasn't redeployed. |
| `backend unreachable` (502) | Modal deploy failed or container is past timeout — check `modal logs`. |
| `analyze failed: 415` | Upload Content-Type isn't a supported video; the frontend already filters by `video/*` so this only happens via direct API hits. |
| "no matches found" in the result | Reference library wasn't seeded into the *cloud* cluster — the local `.qdrant-data` doesn't carry over to Cloud. Re-run `build_library.sh` with cloud credentials. |

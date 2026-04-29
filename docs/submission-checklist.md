# Submission checklist — Qdrant *Think Outside the Bot*

Deadline: **2026-06-01**.

## Before you record the demo video

- [ ] **GitHub remote pushed.** From the repo root:
      ```bash
      gh repo create mimiccoach --public --source=. --remote=origin --push
      ```
- [ ] **Qdrant Cloud cluster up.** Free-tier cluster created at
      <https://cloud.qdrant.io>; URL + API key in hand.
- [ ] **Modal deployed.** `modal token new` once, then
      `modal deploy backend/app.py`. Save the printed asgi URL.
- [ ] **Reference library seeded into Cloud.** This is critical — the
      synthetic baseline lives in Qdrant Cloud, not your local
      `.qdrant-data` directory.
      ```bash
      QDRANT_URL=https://<cluster>.qdrant.io:6333 \
        QDRANT_API_KEY=<key> \
        ./scripts/build_library.sh
      ```
      Verify with: `curl https://<modal>/motions | jq .`
- [ ] **Vercel deployed.** `vercel --prod` after `vercel env add
      MODAL_BACKEND_URL production`.
- [ ] **End-to-end smoke test.** Open the Vercel URL, upload a tennis
      serve clip, see a result page with five phase scores.
- [ ] **Warm-keep is firing.** `modal app list` should show `warm_keep`
      scheduled; the analyze container stays hot.

## What the submission needs

The hackathon rules ask for:

- [ ] **GitHub repo URL** — public, with README at the root.
- [ ] **README** — covers what it does, how it uses Qdrant, how to run.
      Already in shape; check it once more for typos.
- [ ] **Demo video URL** — ≤ 3:00, hosted on YouTube or Vimeo, public.
      See [`demo-storyboard.md`](demo-storyboard.md).
- [ ] **(Optional) Live demo URL** — the Vercel deployment.

## Repo hygiene before submitting

- [ ] LICENSE file present (MIT, already in repo).
- [ ] Dataset attributions visible in the README's "Credits" section.
- [ ] No secrets committed (`.env`, API keys, Modal tokens).
- [ ] `frontend/.env.local` is in `.gitignore` (it is).
- [ ] `pytest` green: `cd backend && uv run pytest` → 76 passed.
- [ ] `pnpm build` green: `cd frontend && pnpm build` → 6 routes built.
- [ ] CI green on the `main` branch (if you set up GitHub Actions runs).

## Day-of-submission

- [ ] Visit the hackathon submission portal.
- [ ] Submit the GitHub URL, demo video URL, and live demo URL.
- [ ] Tag your video with the hackathon hashtag if one is requested.
- [ ] Save a screenshot of the submission confirmation.

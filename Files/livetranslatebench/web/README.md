# LiveTranslateBench — web leaderboard

A static, dependency-free page that visualizes benchmark results. It reads
`leaderboard.json` from this folder and renders a lag ruler (how far each system
trails the speaker) plus a sortable metrics table.

## What deploys here (and what doesn't)

Only this `web/` folder is meant for Vercel. The benchmark **harness itself does
not run on Vercel** — it holds a minutes-long real-time WebSocket stream to the
Gemini Live API and (for baselines) loads multi-gigabyte ML models, neither of
which fits Vercel's short-lived, size-capped serverless functions. Run the
harness locally or in CI; publish the resulting JSON here.

## The data flow

```
run harness (local / CI) ──▶ results/*.jsonl
        │
        ▼
ltbench leaderboard ──▶ web/leaderboard.json   (committed)
        │
        ▼
git push ──▶ Vercel redeploys the static page
```

`leaderboard.json` ships empty, so a fresh deploy shows an honest empty state
with run instructions. Click "Preview with sample data" to see the layout;
sample numbers are clearly watermarked and are never published as real.

## Deploy on Vercel

1. Push the repo to GitHub (see the top-level README).
2. In Vercel: **New Project → import the repo**.
3. Set **Root Directory** to `livetranslatebench/web`.
4. **Framework Preset:** Other. No build command. Output directory: `.` (the
   folder is already static).
5. Deploy. Every later `git push` that updates `leaderboard.json` redeploys.

CLI alternative:

```bash
cd livetranslatebench/web
npx vercel        # preview
npx vercel --prod # production
```

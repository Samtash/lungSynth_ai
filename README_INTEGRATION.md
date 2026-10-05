# LungSynth AI

A web app for the proposal *"Phase-Conditioned Diffusion Models for 4D
Lung CT Synthesis"* (NSU CSE, Spring 2025): upload the two clean
boundary CT phases (T00, T50) and reconstruct the eight intermediate
breathing phases with a diffusion model.

```
lung-phase-synth-main/
  src/            React 19 + TanStack Start frontend (routes: dashboard, processing, results, history, login, settings)
  backend/        FastAPI backend: real 3D U-Net + FiLM + Phase-Aware Attention diffusion model
```

## Quick start

**1. Backend** (terminal 1):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Confirm it's up: `curl http://localhost:8000/api/health`

**2. Frontend** (terminal 2):

```bash
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:3000`). Sign in
(mocked — no real credential check yet, see note below), go to
**Dashboard**, upload two CT volumes as T00/T50, and click **Generate
CT Phases**. This now does real work: it uploads to the backend,
starts an actual diffusion sampling job, polls live progress, and
renders the real generated slice previews + downloadable `.nii.gz`
volumes on the Results page.

If your backend isn't on `localhost:8000`, set `VITE_API_BASE_URL` in
a `.env` file at the repo root before running `npm run dev`.

## What's real vs. what's still a demo

**Real:**
- The 3D U-Net + FiLM + Phase-Aware Attention model (`backend/app/models/`)
- The DDPM training objective and DDIM sampler (`backend/app/core/diffusion.py`)
- DICOM/NIfTI/MetaImage/NRRD loading, HU normalization, resampling (`backend/app/core/preprocessing.py`)
- The full upload → async job → poll → results → download flow between frontend and backend
- Session history, persisted server-side in SQLite

**Still a demo / needs follow-up work:**
- **No trained checkpoint is included.** The proposal itself is pre-training (Section 5.1: *"experimental results have not yet been obtained"*). Until you run `backend/train.py` against the DIR-Lab dataset and drop a checkpoint at `backend/checkpoints/phasediff.pt`, generated volumes are structurally valid but not clinically meaningful. This is explained on the Results page whenever no checkpoint is loaded.
- **Google Sign-In is still mocked** in `src/routes/login.tsx` (sets a fake local user, no network call). `backend/app/routers/auth.py` implements real Google ID token verification and is ready to wire up — see `backend/README.md` for the steps and env vars.
- **History for past sessions can't be re-opened** (click "New Generation" instead) — the in-memory job store doesn't persist generated volumes across a backend restart. Swap the job store for persisted storage if you need this.
- Inference resamples the whole volume to a fixed working cube (default 96³) rather than the proposal's patch-stitched full-resolution pipeline, for demo responsiveness. See `backend/README.md`'s "Known simplifications" section.

See `backend/README.md` for backend-specific configuration, API reference, and architecture details.

# LungSynth AI — Backend

FastAPI backend implementing the phase-conditioned diffusion model from
the proposal *"Phase-Conditioned Diffusion Models for 4D Lung CT
Synthesis"*: a 3D U-Net noise predictor with FiLM conditioning at every
residual block and a Phase-Aware Attention Module at the bottleneck,
trained/sampled as a DDPM with DDIM inference (~50 steps).

This is **real inference**, not a mock — uploads go through actual
preprocessing, the actual model does a real forward pass, and DDIM
actually denoises from Gaussian noise. What it is *not*, out of the
box, is a **trained** model — see the callout below.

## ⚠️ About "real inference" without a trained checkpoint

The source proposal is a *research proposal*: no model has been trained
on the DIR-Lab dataset yet (Section 5.1 says as much: *"Since this is a
proposal, experimental results have not yet been obtained"*). So:

- If `checkpoints/phasediff.pt` doesn't exist, the pipeline runs with
  **randomly initialised weights**. You'll get correctly-shaped,
  CT-window-normalized volumes flowing through a real diffusion
  sampling loop — but the anatomical content is not meaningful yet.
- `train.py` is a full LOOCV training script matching Section 4.3.4 of
  the proposal (patch-based 64³ training, AdamW, cosine annealing, 200
  epochs). Run it against a local copy of the DIR-Lab dataset
  (https://www.dir-lab.com) to produce a real checkpoint, then drop it
  at `checkpoints/phasediff.pt` — the server picks it up automatically
  on next boot (see `/api/health` → `checkpointLoaded`).
- The `confidence` value returned per phase is **not** a calibrated
  clinical score (that requires Target Registration Error against
  DIR-Lab's 300 annotated landmarks — see Section 5.1.4 — evaluated on
  a trained model). It's currently a reconstruction-consistency
  heuristic against the linear-interpolation baseline. Swap in real
  TRE-based confidence once you have a trained + validated model.

## Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The frontend (see `src/lib/api.ts`) expects the backend at
`http://localhost:8000` by default — set `VITE_API_BASE_URL` if you run
it elsewhere.

Health check: `GET http://localhost:8000/api/health`

## Configuration

All settings live in `app/config.py` and can be overridden via
environment variables prefixed `LUNGSYNTH_`, or a `backend/.env` file:

| Variable | Default | Notes |
|---|---|---|
| `LUNGSYNTH_WORKING_SIZE` | `96` | Volumes are resampled to this cubic size before diffusion sampling. Larger = more detail, much slower on CPU. |
| `LUNGSYNTH_DDIM_STEPS` | `50` | Matches the proposal's Section 1.2 point 4 (1000→~50 step speedup). |
| `LUNGSYNTH_CHECKPOINT_PATH` | `checkpoints/phasediff.pt` | Where a trained model is loaded from, if present. |
| `LUNGSYNTH_GOOGLE_CLIENT_ID` | unset | Set this to enable verified Google Sign-In. Unset = dev mode (unverified token decode). |
| `LUNGSYNTH_CORS_ORIGINS` | localhost:8080/3000 | Add your deployed frontend origin here. |

GPU is used automatically if `torch.cuda.is_available()`; otherwise it
runs on CPU (slow but functional — a full 7-phase generation at the
default 96³/50-step settings takes a while on CPU; drop
`LUNGSYNTH_WORKING_SIZE` and `LUNGSYNTH_DDIM_STEPS` for faster local
testing).

## API surface

| Endpoint | Purpose |
|---|---|
| `POST /api/sessions/upload` | multipart upload of `t00` + `t50` scans → `sessionId` |
| `POST /api/sessions/{id}/generate` | kicks off async generation → `jobId` |
| `GET /api/jobs/{jobId}` | poll status/stage/progress (matches the frontend's 5-stage progress screen) |
| `GET /api/sessions/{id}/results` | phase metadata once the job completes |
| `GET /api/sessions/{id}/phases/{label}/preview` | PNG mid-slice preview |
| `GET /api/sessions/{id}/phases/{label}/download` | `.nii.gz` volume download |
| `GET /api/sessions/{id}/download-all` | zip of every phase (volumes + previews) |
| `POST /api/sessions/{id}/save-history` | persists a history row |
| `GET /api/history` / `DELETE /api/history` | session history |
| `POST /api/auth/google` | verifies (or, in dev mode, decodes) a Google ID token |

## Project layout

```
backend/
  app/
    main.py            FastAPI app + startup model warm-load
    config.py           settings
    schemas.py           Pydantic request/response models
    storage.py            SQLite session history
    models/
      unet3d.py            3D U-Net + FiLM residual blocks + Phase-Aware Attention
      embeddings.py          sinusoidal timestep/phase embeddings + FiLM
    core/
      diffusion.py           DDPM forward process, training loss, DDIM sampler
      preprocessing.py        DICOM/NIfTI/MetaImage/NRRD I/O, HU normalize, resample
      inference.py             the generation pipeline (loads model once, runs all 7 phases)
      jobs.py                   async job tracker (status polling for the frontend)
    routers/
      generate.py               upload/generate/results/download endpoints
      history.py                 session history endpoints
      auth.py                     Google Sign-In verification
  train.py                LOOCV training script for the DIR-Lab dataset
  checkpoints/             trained model weights go here
  data/                     uploads/outputs/sqlite db (gitignored)
```

## Known simplifications vs. the full proposal

- **Full-volume resize instead of patch-stitched inference.** The
  proposal trains on 64³ patches with 50%-overlap sliding-window
  stitching for full-resolution (512×512×~120) output. For a
  responsive web demo, `inference.py` instead resamples the whole
  volume to a fixed working cube (`working_size`, default 96³) and runs
  a single diffusion pass. `train.py` *does* implement the paper's
  actual patch extraction for training. If you need patch-stitched
  full-resolution inference for real clinical-grade output, extend
  `PhaseSynthPipeline.generate()` with a sliding-window + averaging
  pass — the model and diffusion code underneath don't need to change.
- **Confidence score** is a heuristic, not TRE — see callout above.
- **Job queue is in-memory**, single-process. Fine for a demo/single
  instance; swap for Redis/Celery if you need multiple workers.

# LungSynth AI Backend

FastAPI backend for the phase conditioned diffusion model. The noise predictor is a 3D UNet with FiLM conditioning in every residual block and a Phase Aware Attention module at the bottleneck. It is trained as a DDPM and sampled with DDIM in about 50 steps.

Every request goes through real preprocessing and a real diffusion sampling loop. Nothing is mocked.

## Model weights

The server loads a trained checkpoint from `checkpoints/phasediff.pt` on startup. You can confirm it loaded at `GET /api/health`, which reports `checkpointLoaded`.

The checkpoint is not in the repo because of file size. If it is missing the pipeline still runs with untrained weights. Shapes and the full flow will work but the output will not look like real anatomy.

To train your own, see `train.py`. It does patch based training on 64³ crops with AdamW and cosine annealing, using leave one out cross validation across the DIR-Lab patients. You need a local copy of DIR-Lab from https://www.dir-lab.com.

## Setup

```bash
cd backend
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

The frontend expects the backend at `http://localhost:8000`. Set `VITE_API_BASE_URL` in the root `.env` if you run it somewhere else.

## Configuration

Settings live in `app/config.py`. Any of them can be overridden with an environment variable that starts with `LUNGSYNTH_`, or in `backend/.env`.

| Variable | Default | Notes |
|---|---|---|
| `LUNGSYNTH_WORKING_SIZE` | `96` | Volumes are resampled to this cube size before sampling. Bigger means more detail but much slower on CPU. |
| `LUNGSYNTH_DDIM_STEPS` | `50` | Number of DDIM sampling steps. |
| `LUNGSYNTH_CHECKPOINT_PATH` | `checkpoints/phasediff.pt` | Where the trained model is loaded from. |
| `LUNGSYNTH_GOOGLE_CLIENT_ID` | not set | Turns on verified Google sign in. If it is not set the server runs in dev mode and does not verify tokens. Never deploy like that. |
| `LUNGSYNTH_CORS_ORIGINS` | localhost:8080 and 3000 | Add your deployed frontend address here. |

A GPU is used automatically if PyTorch can see one. On CPU it still works but a full run is slow. Lower the working size and DDIM steps for quick local tests.

## API

| Endpoint | What it does |
|---|---|
| `POST /api/sessions/upload` | Upload `t00` and `t50` scans, returns a `sessionId` |
| `POST /api/sessions/{id}/generate` | Starts a background generation job, returns a `jobId` |
| `GET /api/jobs/{jobId}` | Job status, stage and progress for the progress screen |
| `GET /api/sessions/{id}/results` | Phase details once the job is done |
| `GET /api/sessions/{id}/phases/{label}/preview` | PNG preview of the middle slice |
| `GET /api/sessions/{id}/phases/{label}/download` | `.nii.gz` volume for one phase |
| `GET /api/sessions/{id}/download-all` | Zip of every phase, volumes and previews |
| `POST /api/sessions/{id}/save-history` | Saves the run to history |
| `GET /api/history` and `DELETE /api/history` | Read or clear history |
| `POST /api/auth/google` | Verifies a Google ID token and returns the user |

## Layout

```
backend/
  app/
    main.py              FastAPI app, loads the model once at startup
    config.py            settings
    schemas.py           Pydantic request and response models
    storage.py           SQLite session history
    models/
      unet3d.py          3D UNet with FiLM residual blocks and Phase Aware Attention
      embeddings.py      sinusoidal timestep and phase embeddings, FiLM
    core/
      diffusion.py       DDPM forward process, training loss, DDIM sampler
      preprocessing.py   DICOM, NIfTI, MetaImage and NRRD loading, HU normalization, resampling
      inference.py       generation pipeline for all seven phases
      jobs.py            background job tracker
    routers/
      generate.py        upload, generate, results and download endpoints
      history.py         history endpoints
      auth.py            Google sign in verification
  train.py               training script for DIR-Lab
  checkpoints/           trained weights go here (not committed)
  data/                  uploads, outputs and the SQLite database (not committed)
```

## Known simplifications

**Single pass inference.** Training uses 64³ patches with overlapping windows. For a responsive web demo, `inference.py` instead resamples the whole volume to one working cube (96³ by default) and runs a single diffusion pass. Full resolution output would need a sliding window pass added to `PhaseSynthPipeline.generate()`. The model and diffusion code would not need to change.

**Confidence score.** The per phase confidence value is a consistency check against a linear interpolation baseline. It is not a calibrated clinical score.

**In memory job queue.** Jobs run in a single process. A multi worker setup would need something like Redis or Celery.

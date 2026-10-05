# LungSynth AI

**Phase Conditioned Diffusion Model for 4D Lung CT Intermediate Phase Reconstruction**

Give LungSynth two CT scans of the lungs taken at the two ends of a breath (T00 and T50). It rebuilds the seven breathing phases in between (T10 to T40 and T60 to T80) with a 3D diffusion model. The whole thing runs inside a web app where you sign in, upload scans, watch the job run and download the results.

This was our final year AI capstone project at North South University.

## Why it matters

4D CT shows how the lungs and any tumor inside them move while a patient breathes. Doctors use it to plan radiotherapy for lung cancer. In practice some phases come out noisy or full of motion artifacts. If you can rebuild the middle phases from the two clean end phases, you get a full breathing cycle you can actually use.

## Results

We tested on the DIR-Lab 4D CT dataset.

| Metric | What we compared | Result |
|---|---|---|
| TRE (lower is better) | LungSynth vs no motion baseline | **3.85 mm** vs 4.50 mm |
| MSE (lower is better) | Phase Aware Attention model vs baseline UNet | **about 70% lower** |

TRE (Target Registration Error) checks how far the expert annotated landmarks in DIR-Lab end up from where they should be, in millimeters. It is the clinical metric so we treat it as the main one.

One honest note. A simple linear blend of the two input scans scored better than our model on MSE. That is a known weakness of MSE because it rewards blurry, averaged images. TRE does not have that problem, which is why we lead with it.

## How it works

**Model.** A 3D UNet that predicts noise. FiLM layers inside every residual block feed in the target phase and the diffusion timestep, so one network can produce any of the seven phases. A Phase Aware Attention module sits at the bottleneck.

**Training data.** DIR-Lab only has a handful of patients. To get more to learn from, we used ANTsPy SyN deformable registration on the DIR-Lab landmarks to make extra intermediate phase volumes.

**Training.** Standard DDPM objective on 64³ patches with AdamW (learning rate 2e-4) and cosine annealing. See `backend/train.py`.

**Sampling.** DDIM with 50 steps instead of the full 1000, which makes generation fast enough for a web app.

## The web app

Sign in with Google. Upload a T00 and a T50 scan in DICOM, NIfTI, MetaImage or NRRD format. The backend runs the diffusion job in the background while the frontend shows live progress. When it is done you get a preview slice for every phase. You can download each phase as a `.nii.gz` volume or grab all of them in one zip. Past runs show up on the History page.

<!--
Screenshots: put images in a docs/ folder and remove these comment lines.
![Dashboard](docs/dashboard.png)
![Results](docs/results.png)
-->

## Tech stack

| Part | Tools |
|---|---|
| Model | PyTorch, 3D UNet, FiLM conditioning, DDPM training, DDIM sampling |
| Medical imaging | SimpleITK, nibabel, ANTsPy (training data prep) |
| Backend | FastAPI, SQLite, Google ID token verification |
| Frontend | React 19, TypeScript, TanStack Start, Tailwind CSS, shadcn/ui |

## Run it locally

You need Python 3.10 or newer and Node 20 or newer.

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add your Google client ID
uvicorn app.main:app --reload --port 8000
```

Check it is up at `http://localhost:8000/api/health`.

**Model weights.** The trained checkpoint is not in this repo because of file size. Place it at `backend/checkpoints/phasediff.pt` and the server loads it on startup. Without it the full pipeline still runs, but on untrained weights, so the output will not look like real anatomy. Weights are available on request.

### 2. Frontend

In a second terminal, from the repo root:

```bash
npm install
cp .env.example .env      # add the same Google client ID
npm run dev
```

Open the local address that Vite prints.

### 3. Training (optional)

Download the DIR-Lab dataset from https://www.dir-lab.com first, then:

```bash
cd backend
python train.py --data-dir data/dirlab --epochs 200 --out checkpoints/phasediff.pt
```

## Project layout

```
src/                  React frontend (login, dashboard, processing, results, history, settings)
backend/
  app/models/         3D UNet, FiLM blocks, Phase Aware Attention
  app/core/           diffusion, preprocessing, inference pipeline, job queue
  app/routers/        API endpoints for generation, history and auth
  train.py            training script
```

More detail on the API and settings is in [backend/README.md](backend/README.md).

## Limitations

The app runs one pass on a 96³ resampled volume to keep things quick. It does not stitch full resolution patches the way a clinical tool would. The confidence score shown for each phase is a consistency check, not a clinical measure. The job queue lives in memory so it runs as a single process.

This is a research project. It is not meant for diagnosis or treatment decisions.

## Team

Built by a team of CSE students at North South University as our final year capstone.

##Demo

https://github.com/user-attachments/assets/201ad84a-7a1b-4b52-b38c-f52bc0b41c65



# LungSynth AI

LungSynth AI is an AI-powered web application designed to generate intermediate 4D lung CT scan phases from two boundary CT scans. The application provides a simple, intuitive interface that allows users to upload medical images, process them using an AI model, and visualize the generated intermediate phases.

---

## Overview

LungSynth AI assists researchers and medical professionals by reconstructing missing respiratory phases between two CT scans. The system is intended to support research in medical imaging and respiratory motion analysis by providing high-quality AI-generated intermediate images.

---

## Features

- Secure Google Authentication
- Upload two CT scan images (Boundary Phases)
- AI-powered intermediate phase generation
- Processing progress indicator
- View generated CT phases
- Download generated results
- History of previous generations
- User profile and settings management
- Responsive and modern medical dashboard

---

## Workflow

1. Launch the application.
2. Sign in using your Google account.
3. Upload the required CT scan images.
4. Click **Generate CT Phases**.
5. Wait while the AI model processes the images.
6. View the generated intermediate CT phases.
7. Download the generated images if required.
8. Access previous generations from the History page.

---

## Application Pages

### Login

Secure authentication using Google Sign-In.

### Upload

Upload the required CT scan images for processing.

### Processing

Displays the current processing status while the AI model generates intermediate phases.

### Results

Displays all generated CT scan phases with options to enlarge or download images.

### History

Stores previous image generation sessions along with upload date and time.

### Settings

Allows users to:

- Edit display name
- Change preferred date format
- Change preferred time format
- Logout

---

## Technologies Used

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- TanStack Router
- Shadcn UI
- Lucide Icons

### Backend

- Python
- FastAPI

### AI Model

- Diffusion Model
- Medical Image Processing

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/lungsynth-ai.git
cd lungsynth-ai
```

### Install dependencies

```bash
npm install
```

### Start the development server

```bash
npm run dev
```

The application will be available at:

```
http://localhost:8080
```

---

## Project Structure

```
src/
 ├── components/
 ├── hooks/
 ├── lib/
 ├── routes/
 ├── router.tsx
 ├── start.ts
 └── styles.css

public/
 ├── favicon.ico
 └── robots.txt
```

---

## Future Improvements

- Support for additional medical image formats
- 3D CT volume visualization
- Patient report generation
- AI confidence visualization
- Multi-user collaboration
- Cloud storage integration

---

## Disclaimer

This application is intended for research and educational purposes. It is not designed to replace professional medical diagnosis or clinical decision-making.

---

## License

This project is intended for academic and research use.
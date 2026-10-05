

# LungSynth AI

**LungSynth AI** is an advanced medical web application designed to synthesize intermediate 4D lung CT scan phases from two boundary scans. Featuring a modern UI, the platform allows researchers to upload scans, run AI inference, and visualize dynamic respiratory motion.

> **Note:** This software is developed strictly for research and academic purposes and should not be used as a substitute for clinical medical diagnosis.
> 
> 

---

## 🌟 Key Features

* **Google Authentication:** Secure login via Google credentials.


* **4D Synthesis:** Upload boundary CT images to generate missing intermediate phases.


* **Real-time Monitoring:** Interactive progress indicators during model processing.


* **Visualization & Export:** Inspect, zoom, and download generated CT frames.


* **Session History:** Access and review past generation workflows.


* **Customization:** Configurable user profile settings, including date and time display options.



---

## 🔄 App Workflow

1. **Sign In:** Authenticate using your Google account.


2. **Upload:** Select and upload boundary CT scan images.


3. **Generate:** Trigger the AI pipeline and monitor real-time execution.


4. **Analyze & Export:** View intermediate scan frames or download outputs.


5. **History:** Revisit past processing sessions.



---

## 🛠️ Tech Stack

* **Frontend:** React, TypeScript, Vite, Tailwind CSS, TanStack Router, Shadcn UI, Lucide Icons


* **Backend:** Python, FastAPI


* **AI Engine:** Diffusion Models, Medical Image Processing Libraries



---

## 🚀 Getting Started

### Prerequisites

* Node.js (v18 or higher recommended)
* npm or yarn

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/lungsynth-ai.git
cd lungsynth-ai

```


2. **Install dependencies:**
```bash
npm install

```


3. **Run the development server:**
```bash
npm run dev

```



Access the application in your browser at `http://localhost:8080`.

---

## 📁 Repository Structure

```text
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

## 🗺️ Roadmap & Future Enhancements

* Expansion to additional medical file formats (e.g., DICOM)


* 3D volume reconstruction and rendering


* Automated patient report generation


* AI model confidence and uncertainty visualization


* Cloud integration & multi-user collaboration



---

## 📜 License

This project is open-source and intended for academic, educational, and research use.

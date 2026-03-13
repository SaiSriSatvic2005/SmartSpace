---
title: SmartSpace Studio
emoji: ""
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Privacy-first layout analysis using CV, spatial math, RAG.
---

# SmartSpace Studio

SmartSpace Studio is a privacy-first interior layout optimization system built as a standalone web app. It combines computer vision, deterministic spatial reasoning, and Retrieval-Augmented Generation (RAG) to produce actionable, architecture-aligned layout recommendations from real room images.

## Key Features
- Multi-image analysis (1 to 3 photos) for robust detection across angles.
- Spatial reasoning with zone density, collision detection, and coverage metrics.
- Room utilization heatmap overlay and a 0 to 100 Space Efficiency Score.
- Preference-aware recommendations (room type, style, and priority).
- In-app PDF knowledge upload with on-demand RAG index rebuild.
- Fully local inference using TinyLlama via ctransformers.

## Tech Stack
- Frontend: Streamlit
- Vision: Ultralytics YOLOv8m
- RAG: LangChain ecosystem, FAISS, all-MiniLM-L6-v2 embeddings
- LLM: TinyLlama 1.1B quantized GGUF via ctransformers
- Image processing: Pillow, NumPy
- PDF ingestion: PyPDF2
- Deployment: Docker on Hugging Face Spaces

## Local Setup
1. Create and activate a virtual environment.
2. Install dependencies:
	pip install -r requirements.txt
3. Run the app:
	streamlit run app.py

The app will download model weights on first run if they are not present in the working directory.

## Standalone Deployment (Hugging Face Spaces)
This project is deployed as a Docker Space. The container runs Streamlit on port 7860.

Core deployment steps:
1. Create a Docker Space.
2. Push source code and Dockerfile (do not commit model binaries).
3. Let the app download models at runtime.
4. If image uploads fail behind the proxy, set Streamlit server config in .streamlit/config.toml.

Live URL:
https://huggingface.co/spaces/nonchok005/SamrtSpace-Studio

## Usage Flow
1. Upload 1 to 3 room images.
2. Choose room preferences.
3. Run spatial analysis to see heatmap and metrics.
4. Upload PDFs to expand knowledge (optional).
5. Click Rebuild RAG Knowledge Base.
6. Read the final layout recommendations.

## Privacy Notes
The system is designed to run locally or on a standalone deployment without relying on third-party inference APIs. Images are processed inside the app runtime and are not forwarded to external LLM services.

## Limitations
- 2D-only spatial modeling (no true depth or volumetric reasoning).
- Detection accuracy depends on lighting, occlusion, and camera angle.
- CPU-only inference limits model size and context length.

## License
MIT

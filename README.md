# WasteWise Waste Classification System

WasteWise is a full-stack waste classification app that uses a TensorFlow model to classify uploaded waste images as recyclable or non-recyclable.

## Repository Structure

```text
WasteWise/
├── frontend/
│   ├── public/
│   ├── src/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── ...
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── waste_classifier.keras
│   ├── uploads/
│   └── ...
├── README.md
└── .gitignore
```

## Features

- Image upload with preview before prediction
- FastAPI backend with multipart upload support
- TensorFlow model inference for recyclable vs non-recyclable waste
- Confidence score returned from the backend
- Dark theme default on the frontend
- Reupload flow after each classification
- CORS support for frontend-backend communication

## Tech Stack

- Frontend: Vite, vanilla JavaScript, HTML, CSS
- Backend: FastAPI, Uvicorn
- ML: TensorFlow, NumPy, Pillow
- Deployment: Vercel for frontend, Render for backend

## Installation

### Prerequisites

- Python 3.10+ recommended
- Node.js 18+
- npm

### Clone the repository

```bash
git clone https://github.com/anshux-me/WasteWise.git
cd WasteWise
```

## Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Backend environment variables:

- `CORS_ORIGINS` - comma-separated list of allowed frontend origins

Example local defaults:

```bash
export CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend environment variables:

- `VITE_API_BASE_URL` - backend base URL, for example `http://127.0.0.1:8000`

Example local defaults:

```bash
export VITE_API_BASE_URL="http://127.0.0.1:8000"
```

## Running Locally

1. Start the backend from the `backend/` folder.
2. Start the frontend from the `frontend/` folder.
3. Open the frontend URL shown by Vite.

## API Documentation

### Health Check

`GET /health`

Response:

```json
{ "status": "healthy" }
```

### Predict Waste Type

`POST /predict`

Content type: `multipart/form-data`

Form field:

- `file`: image file

Example response:

```json
{
  "prediction": "Recyclable",
  "confidence": 92.45
}
```

Example response for a non-recyclable item:

```json
{
  "prediction": "Non-Recyclable",
  "confidence": 87.12
}
```

## Screenshots

Add screenshots here after deployment:

- Frontend home screen
- Upload preview state
- Prediction result screen
- Mobile responsive view

## Deployment Instructions

### Deploy Backend on Render

1. Create a new Render Web Service.
2. Point it to this repository.
3. Set the service root directory to `backend`.
4. Add environment variables as needed:
   - `CORS_ORIGINS=https://your-vercel-domain.vercel.app`
5. Use this start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Deploy Frontend on Vercel

1. Create a new Vercel project from the same repository.
2. Set the root directory to `frontend`.
3. Use these build settings:
   - Build command: `npm run build`
   - Output directory: `dist`
4. Add the environment variable:
   - `VITE_API_BASE_URL=https://your-render-backend.onrender.com`

## Notes

- The backend returns the model confidence as a percentage.
- The frontend reupload button lets users analyze another image after each result.
- Keep `backend/waste_classifier.keras` in place for deployment.
- The repository is now organized for deployment from `frontend/` and `backend/` only.

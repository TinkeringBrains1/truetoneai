import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2Model
from huggingface_hub import hf_hub_download

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================
# Repo details for fetching the fine-tuned model weights
REPO_ID = "TaterTots123/human.ai"
FILENAME = "v1.pth"  # Ensure this matches your uploaded weight file

print(f"⬇️  Downloading model weights from {REPO_ID}...")
try:
    # Downloads the model file to the local HuggingFace cache
    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"✅ Model downloaded to: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    # In production, you might want to exit here if the model is missing

# Select GPU if available, otherwise CPU (Auto-detection)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️  Running on device: {DEVICE}")

# Secure API Key from Environment Variables
SECRET_KEY = os.getenv("API_KEY") 
if not SECRET_KEY:
    print("⚠️  WARNING: API_KEY not found! API is unsecured.")

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
# This class defines the custom Classification Head (Backend) used in AASIST.
# It takes the raw features from Wav2Vec2 and decides "Fake" or "Real".
class AASIST_Backend(nn.Module):
    def __init__(self, in_channels=128, emb_dim=128):
        super().__init__()
        # Convolutional layers to extract spatial/temporal patterns
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Self-Attention mechanism to focus on the most "suspicious" parts of audio
        self.attention = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        
        # Final classification layers
        self.fc = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, 2) # Output: [Human_Score, AI_Score]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.squeeze(-1).transpose(1, 2)
        
        # Attention pooling
        w = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(w * x, dim=1)
        
        x = F.relu(self.fc(x))
        return self.classifier(x)

# This is the main container class combining the Backbone + Backend
class VoiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        print("🌍 Initializing MMS-300M Backbone...")
        # Load the pre-trained Facebook MMS model (Raw Audio Feature Extractor)
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/mms-300m")
        
        # Projection layer to match dimensions between Backbone and Backend
        self.proj = nn.Linear(1024, 128) 
        self.backend = AASIST_Backend()

    def forward(self, wav_input):
        # 1. Extract features using Wav2Vec2 (Gradient disabled for backbone efficiency)
        with torch.no_grad():
            outputs = self.backbone(wav_input)
            features = outputs.last_hidden_state
        
        # 2. Project and classify
        x = self.proj(features).transpose(1, 2).unsqueeze(-1)
        return self.backend(x)

# ==============================================================================
# 3. SERVER INITIALIZATION
# ==============================================================================
app = FastAPI(title="Deepfake Voice Detection API")

# Enable CORS to allow requests from any website/app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model into memory once on startup
print(f"⏳ Loading Weights onto {DEVICE}...")
model = VoiceDetector().to(DEVICE)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode (disable dropout, etc.)
    print("✅ Model Weights Loaded Successfully!")
except Exception as e:
    print(f"❌ Critical Error loading weights: {e}")

# ==============================================================================
# 4. PREPROCESSING PIPELINE
# ==============================================================================
def preprocess_basic(wav):
    """
    Standard Z-Score Normalization (Standardization).
    
    Formula: x_norm = (x - mean) / std
    
    Why?
    Neural networks work best when inputs are centered around 0 with unit variance.
    This removes volume bias (e.g., quiet files shouldn't be classified differently).
    """
    if len(wav) > 0:
        mean = np.mean(wav)
        std = np.std(wav) + 1e-9 # Add epsilon to prevent division by zero
        wav = (wav - mean) / std
    return wav

# ==============================================================================
# 5. API ENDPOINTS
# ==============================================================================

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Voice Detection API is live. Accepts .wav and .mp3 only."
    }

class AudioRequest(BaseModel):
    audioBase64: str # The only required input

@app.post("/api/voice-detection")
async def detect_voice(req: AudioRequest, x_api_key: str = Header(None)):
    """
    Main Inference Endpoint.
    1. Validates API Key.
    2. Decodes Base64 Audio.
    3. Resamples to 16kHz & Mono.
    4. Normalizes.
    5. Returns AI Probability.
    """
    
    # --- Step 1: Security Check ---
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # --- Step 2: Decode Base64 ---
        try:
            # Strip header if present (e.g., "data:audio/mp3;base64,...")
            if "base64," in req.audioBase64:
                req.audioBase64 = req.audioBase64.split("base64,")[1]
            audio_bytes = base64.b64decode(req.audioBase64)
        except Exception:
            return {"status": "error", "message": "Invalid Base64 string"}

        # --- Step 3: Load Audio (Librosa) ---
        # Librosa automatically:
        # - Decodes WAV/MP3 from memory bytes
        # - Resamples to 16000 Hz (Critical for Wav2Vec2)
        # - Mixes stereo to mono
        try:
            wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            print(f"Audio Load Error: {e}")
            return {"status": "error", "message": "Invalid audio format. Please use WAV or MP3."}

        # --- Step 4: Preprocessing (Normalize) ---
        wav = preprocess_basic(wav)

        # --- Step 5: Model Inference ---
        # Convert numpy array to PyTorch Tensor
        tensor_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(): # Disable gradient calculation for speed
            logits = model(tensor_wav)
            probs = torch.softmax(logits, dim=1)
            
            # Extract probabilities
            human_score = probs[0][0].item()
            ai_score = probs[0][1].item() # Index 1 is the 'Fake/AI' class

        # --- Step 6: Formulate Response ---
        # Strict logic: > 0.5 is AI, otherwise Human
        is_ai = ai_score > 0.50
        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        # Use the dominant score as confidence
        confidence = ai_score if is_ai else human_score

        return {
            "status": "success",
            "classification": classification,
            "confidenceScore": round(confidence, 2)
        }

    except Exception as e:
        print(f"Global Error: {e}")
        return {"status": "error", "message": str(e)}

# Entry point for local debugging
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

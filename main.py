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
import scipy.signal as signal
from transformers import Wav2Vec2Model
from huggingface_hub import hf_hub_download

# ==========================================
# 1. CONFIGURATION
# ==========================================
REPO_ID = "TaterTots123/human.ai"
FILENAME = "v1.pth"

print(f"Downloading model from {REPO_ID}...")
try:
    MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    print(f"Model downloaded to: {MODEL_PATH}")
except Exception as e:
    print(f"Failed to download model: {e}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get API Key
SECRET_KEY = os.getenv("API_KEY") 
if not SECRET_KEY:
    print("⚠️ WARNING: API_KEY not found in environment variables!")

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class AASIST_Backend(nn.Module):
    def __init__(self, in_channels=128, emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.attention = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.fc = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.squeeze(-1).transpose(1, 2)
        w = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(w * x, dim=1)
        x = F.relu(self.fc(x))
        return self.classifier(x)

class VoiceDetector(nn.Module):
    def __init__(self):
        super().__init__()
        print("🌍 Initializing MMS-300M Backbone...")
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/mms-300m")
        self.proj = nn.Linear(1024, 128) 
        self.backend = AASIST_Backend()

    def forward(self, wav_input):
        with torch.no_grad():
            outputs = self.backbone(wav_input)
            features = outputs.last_hidden_state
        x = self.proj(features).transpose(1, 2).unsqueeze(-1)
        return self.backend(x)

# ==========================================
# 3. SERVER SETUP
# ==========================================
app = FastAPI()

# ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"⏳ Loading Model onto {DEVICE}...")
model = VoiceDetector().to(DEVICE)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==========================================
# 4. PREPROCESSING HELPER
# ==========================================
def preprocess_tri_series(wav, sr=16000):
    """
    Applies: Bandpass Filter (70Hz-8kHz) -> Z-Score Normalization
    """
    # 1. Bandpass Filter
    if len(wav) > 0:
        sos = signal.butter(6, [70, 7999], btype='bandpass', fs=sr, output='sos')
        wav = signal.sosfilt(sos, wav)

    # 2. Z-Score Normalization
    if len(wav) > 0:
        mean = np.mean(wav)
        std = np.std(wav) + 1e-9
        wav = (wav - mean) / std
        
    return wav

# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Voice Detection API is live. Accepts .wav and .mp3 only."
    }

class AudioRequest(BaseModel):
    audioBase64: str

@app.post("/api/voice-detection")
async def detect_voice(req: AudioRequest, x_api_key: str = Header(None)):
    # 1. Validate API Key
    if x_api_key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 2. Decode Base64
        try:
            if "base64," in req.audioBase64:
                req.audioBase64 = req.audioBase64.split("base64,")[1]
            audio_bytes = base64.b64decode(req.audioBase64)
        except Exception:
            return {"status": "error", "message": "Invalid Base64 string"}

        # 3. Direct Load (WAV/MP3 only)
        try:
            wav, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            print(f"Audio Load Error: {e}")
            return {"status": "error", "message": "Invalid audio format. Please use WAV or MP3."}

        # 4. Apply Tri-Series Preprocessing
        wav = preprocess_tri_series(wav)

        # 5. Inference (Single Pass - No TTA)
        tensor_wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(tensor_wav)
            probs = torch.softmax(logits, dim=1)
            
            human_score = probs[0][0].item()
            ai_score = probs[0][1].item() # Index 1 = AI Score

        # 6. Classification Logic
        is_ai = ai_score > 0.50
        classification = "AI_GENERATED" if is_ai else "HUMAN"
        confidence = ai_score if is_ai else human_score

        # 7. Return Strictly Filtered Response
        return {
            "status": "success",
            "classification": classification,
            "confidenceScore": round(confidence, 2)
        }

    except Exception as e:
        print(f"Global Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

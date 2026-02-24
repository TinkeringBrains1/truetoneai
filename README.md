# TrueTone AI Voice & Deepfake Detector API

A lightweight, high-performance **FastAPI** service for detecting
**AI-generated, synthetic, and deepfake audio** with state-of-the-art
deep learning models.

Designed for **real-time fraud detection**, **voice authentication**,
and **security research** use cases.

------------------------------------------------------------------------

## Key Features

- Detects **AI-generated vs Human speech**\
- High-performance **PyTorch inference pipeline**\
- Advanced **signal preprocessing & augmentation**\
- Secure API with **API Key authentication**\
- Dynamic model loading from **Hugging Face**\
- Minimal dependencies --- no FFmpeg or heavy audio stacks\
- Production-ready FastAPI architecture

------------------------------------------------------------------------

## Core Architecture

The system combines a **self-supervised speech representation model**
with a **graph-attention anti-spoofing classifier**.

### Feature Extractor Backbone

-   **Facebook MMS-300M (Wav2Vec2)** model
-   Provides rich multilingual speech embeddings
-   Robust against noise and channel variations

###  Backend Classifier

-   **Custom AASIST (Audio Anti-Spoofing using Integrated
    Spectro-Temporal Graph Attention Networks)**
-   Graph-attention based architecture specialized for deepfake
    detection

------------------------------------------------------------------------

## Preprocessing Pipeline

A carefully designed signal processing chain improves robustness:

1.  **Resampling** → 16 kHz mono
2.  **Butterworth Bandpass Filter** → 70 Hz -- 8 kHz
3.  **Z-Score Normalization**

This removes environmental artifacts and focuses the model on human
vocal characteristics.

------------------------------------------------------------------------

### 2-Way Test-Time Augmentation (TTA)

The system performs inference twice:

-   Clean audio
-   White-noise injected variant

Final score = **average of both predictions**

Prevents overfitting to digital silence\
Improves robustness against adversarial audio

------------------------------------------------------------------------

## Tech Stack

-   **Python**
-   **FastAPI**
-   **PyTorch**
-   **Librosa**
-   **Hugging Face Transformers**

------------------------------------------------------------------------

## Installation & Setup

### Clone Repository

``` bash
git clone https://github.com/yourusername/ai-voice-detector-api.git
cd ai-voice-detector-api
```

------------------------------------------------------------------------

### Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### Set Environment Variable (API Key)

Linux / Mac:

``` bash
export API_KEY=your_secret_key
```

Windows (PowerShell):

``` powershell
setx API_KEY "your_secret_key"
```

------------------------------------------------------------------------

### Run Server

``` bash
python main.py
```

Server will start at:

    http://0.0.0.0:7860

------------------------------------------------------------------------

## API Documentation

### Endpoint

    POST /api/voice-detection

------------------------------------------------------------------------

### Headers

    x-api-key: <API_KEY>
    Content-Type: application/json

------------------------------------------------------------------------

### Request Body

``` json
{
  "audioBase64": "<base64_encoded_audio>"
}
```

Supported formats:

-   `.wav`
-   `.mp3`

------------------------------------------------------------------------

### Response

``` json
{
  "status": "success",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.9842
}
```

Classification values:

-   `HUMAN`
-   `AI_GENERATED`

------------------------------------------------------------------------

##  Security

-   API key authentication required
-   No file uploads stored on server
-   Stateless inference pipeline

------------------------------------------------------------------------

##  Use Cases

-   Voice fraud detection
-   Biometric authentication security
-   Deepfake forensics
-   Call screening applications
-   Research & experimentation

------------------------------------------------------------------------

## Performance Goals

-   Low latency inference
-   CPU compatible deployment
-   Scalable containerization ready
-   Minimal memory footprint

------------------------------------------------------------------------

## Project Structure
    ├── main.py
    ├── model/
    ├── utils/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Attribution

This project builds upon the work of leading research organizations:

-   **Meta AI** --- for the **MMS-300M (Wav2Vec2)** speech
    representation model\
    https://ai.meta.com/

-   **Clova AI Research (Naver)** --- for the **AASIST architecture**\
    https://clova.ai/

If you use this project in research, please consider citing their work.

------------------------------------------------------------------------

## Disclaimer

This tool is intended for **research and security purposes only**.

Detection accuracy may vary depending on:

-   Audio quality
-   Compression artifacts
-   Novel synthesis techniques

Always combine with additional verification methods for critical
applications.

------------------------------------------------------------------------

## Contributing

Pull requests are welcome!

For major changes, please open an issue first to discuss what you would
like to change.

------------------------------------------------------------------------

## License

MIT License

------------------------------------------------------------------------

## Author

Developed with care for AI Security & Fraud Prevention.


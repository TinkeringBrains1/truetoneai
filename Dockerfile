# Use Python 3.9 image
FROM python:3.9

# Set working directory
WORKDIR /code

# Install system dependencies (Required for Librosa/Soundfile)
RUN apt-get update && apt-get install -y libsndfile1

# Install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application (including the model)
COPY . .

# Create a cache directory for Transformers and set permissions
RUN mkdir -p /code/cache && chmod 777 /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Run the application on port 7860 (Required by HF Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
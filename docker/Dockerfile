ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.10-py3
FROM ${FROM_IMAGE_NAME}


RUN apt-get update && apt-get install -y libsndfile1 && apt-get install -y sox && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mongolian-speech-recognition

# Install requirements (do this first for better caching)
COPY requirements.txt .
RUN pip install --disable-pip-version-check -U -r requirements.txt

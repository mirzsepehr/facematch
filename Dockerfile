FROM python:3.12-slim

WORKDIR /facematch

COPY ./requirements.txt /facematch/requirements.txt

ENV DEBIAN_FRONTEND noninteractive
# ENV http_proxy="http://127.0.0.1:1080"
# ENV https_proxy="http://127.0.0.1:1080"
# ENV no_proxy="localhost,127.0.0.1,::1"

RUN apt-get clean && apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size
RUN pip3 install --default-timeout=100 -r requirements.txt --no-cache-dir
RUN pip3 install --no-cache-dir uvicorn

COPY ./app /facematch/app 
COPY ./face_detection_yunet_2022mar.onnx /facematch/app/yunet.onnx

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
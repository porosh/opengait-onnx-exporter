FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN pip install --upgrade pip

RUN pip install \
    torch torchvision \
    onnx onnxruntime onnxscript \
    huggingface_hub \
    pyyaml numpy \
    tqdm tensorboard \
    einops kornia matplotlib \
    opencv-python-headless \
    imageio scikit-learn

CMD ["bash"]

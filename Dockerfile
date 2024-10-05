FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Cài đặt các phụ thuộc hệ thống và thiết lập liên kết cho python3.10
RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    libffi7 \
    libp11-kit0 \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt pip và các gói Python
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Clone repository GFPGAN
RUN git clone https://github.com/TencentARC/GFPGAN.git /app/GFPGAN

# Thiết lập thư mục làm việc chính
WORKDIR /app/GFPGAN

# Cài đặt các phụ thuộc Python
RUN pip install basicsr facexlib realesrgan
RUN pip install -r requirements.txt

# Cài đặt GFPGAN ở chế độ phát triển
RUN python setup.py develop

# Tải mô hình pretrained
RUN mkdir -p experiments/pretrained_models/
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models/
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py
# CMD ["python", "checkcuda.py"]
# Định nghĩa lệnh chạy mặc định
# ENTRYPOINT ["python", "inference_gfpgan.py", "-i", "images", "-o", "outputs", "-v", "1.3", "-s", "2"]
EXPOSE 8000

# Run the application using Uvicorn
CMD ["uvicorn", "api_gfpgan:app", "--host", "0.0.0.0", "--port", "8000"]

# 🧛‍♂️ POSTØN Space - Dockerfile com ComfyUI para Hugging Face
FROM python:3.12-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Instalar Node.js 22 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Configurar diretório de trabalho
WORKDIR /app

# Instalar dependências Python
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Remover login desnecessário do Docker Registry

# Instalar ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git comfyui

# Instalar dependências do ComfyUI
RUN cd comfyui && pip install -r requirements.txt

# Copiar código da aplicação
COPY --chown=user app.py .
COPY --chown=user comfyui_client.py .

# Baixar modelo LoRA treinado
RUN mkdir -p comfyui/models/loras
RUN cp lora_training/outputs/final/adapter_model.safetensors comfyui/models/loras/poston_lora.safetensors

# Baixar modelo base SDXL
RUN mkdir -p comfyui/models/checkpoints
RUN wget -O comfyui/models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Baixar VAE
RUN mkdir -p comfyui/models/vae
RUN wget -O comfyui/models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors

# Configurar ComfyUI
RUN echo '{"enable_cors_header": "*"}' > comfyui/extra_model_paths.yaml

# Build do frontend
WORKDIR /app/frontend
COPY --chown=user frontend/package.json ./
RUN npm install --legacy-peer-deps --force
COPY --chown=user frontend/ ./
RUN npm run build

# Voltar para diretório raiz e copiar dist
WORKDIR /app
RUN cp -r ./frontend/dist ./dist

# Script de inicialização
RUN echo '#!/bin/bash\n\
cd comfyui && python main.py --listen 0.0.0.0 --port 8188 &\n\
cd .. && uvicorn app:app --host 0.0.0.0 --port 7860' > start.sh
RUN chmod +x start.sh

# Expor porta
EXPOSE 7860

# Comando de inicialização
CMD ["./start.sh"]
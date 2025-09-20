# 🧛‍♂️ POSTØN Space - Dockerfile simplificado para Hugging Face
FROM python:3.9-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Configurar diretório de trabalho
WORKDIR /app

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app.py .

# Criar arquivo .env
RUN echo "HF_API_TOKEN=your_huggingface_token_here" > .env && \
    echo "PORT=7860" >> .env && \
    echo "NODE_ENV=development" >> .env

# Build do frontend
COPY frontend/package*.json ./
RUN npm install --production
COPY frontend/ ./
RUN npm run build

# Copiar dist para diretório raiz
RUN cp -r ./dist /app/dist

# Expor porta
EXPOSE 7860

# Comando de inicialização
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
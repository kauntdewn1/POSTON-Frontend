# 🧛‍♂️ POSTØN Space - Dockerfile para Hugging Face Spaces
FROM python:3.9

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY app.py .
COPY .env.example ./.env

# Configurar diretório do frontend
WORKDIR /app/frontend

# Instalar Node.js para build do frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Copiar package.json do frontend
COPY frontend/package*.json ./

# Instalar dependências do frontend
RUN npm install

# Copiar código do frontend
COPY frontend/ ./

# Build do frontend
RUN npm run build

# Voltar para diretório raiz
WORKDIR /app

# Copiar arquivos estáticos do frontend buildado
RUN cp -r ./frontend/dist ./dist

# Expor porta
EXPOSE 7860

# Comando de inicialização
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
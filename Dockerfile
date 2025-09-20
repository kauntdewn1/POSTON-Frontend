# 🧛‍♂️ POSTØN Space - Dockerfile para Hugging Face Spaces
FROM node:18-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar package.json e package-lock.json
COPY package*.json ./

# Instalar dependências do backend
RUN npm install

# Copiar código do backend
COPY server.js ./
COPY .env.example ./.env

# Configurar diretório do frontend
WORKDIR /app/frontend

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

# Copiar arquivos estáticos
COPY --from=frontend /app/frontend/dist ./dist

# Expor porta
EXPOSE 7860

# Comando de inicialização
CMD ["node", "server.js"]

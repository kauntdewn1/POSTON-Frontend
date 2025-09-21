#!/bin/bash

# 🧛‍♂️ POSTØN LoRA Caption Generator - Script de Execução
echo "🧛‍♂️ POSTØN LoRA Caption Generator"
echo "=================================="

# Navegar para o diretório do script
cd "$(dirname "$0")"

# Verificar se Python está disponível
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Instalando..."
    exit 1
fi

# Verificar se as pastas existem
if [ ! -d "../data/train" ]; then
    echo "❌ Pasta de treinamento não encontrada: ../data/train"
    exit 1
fi

# Criar pasta de legendas se não existir
mkdir -p "../data/captions"

# Contar imagens
image_count=$(find ../data/train -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l)
echo "📊 Encontradas $image_count imagens para processar"

# Executar o gerador de legendas
echo "🚀 Iniciando geração de legendas..."
python3 generate_captions.py

# Verificar resultados
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Processamento concluído com sucesso!"
    echo "📁 Legendas salvas em: ../data/captions/"
    echo ""
    echo "📋 Arquivos gerados:"
    ls -la ../data/captions/*.txt 2>/dev/null | wc -l | xargs echo "   - Arquivos .txt:"
    echo ""
    echo "🔍 Exemplo de legenda gerada:"
    if [ -f "../data/captions/IMG_9787.txt" ]; then
        echo "   IMG_9787.txt:"
        head -1 ../data/captions/IMG_9787.txt | sed 's/^/     /'
    fi
else
    echo "❌ Erro durante o processamento"
    exit 1
fi

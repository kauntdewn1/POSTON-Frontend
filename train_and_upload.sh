#!/bin/bash
# 🧛‍♂️ POSTØN VISUAL SYSTEM - Script completo de treinamento e upload

echo "🧛‍♂️ Iniciando treinamento do modelo NΞØ Designer v.2050..."

# 1. Instalar dependências
echo "📦 Instalando dependências..."
pip install diffusers transformers accelerate datasets torch torchvision huggingface_hub

# 2. Treinar modelo
echo "🎨 Treinando modelo..."
python train_text_to_image.py \
  --dataset_name ./dataset.jsonl \
  --model_name_or_path runwayml/stable-diffusion-v1-5 \
  --output_dir ./n3o-model \
  --resolution 512 \
  --train_batch_size 4 \
  --num_train_epochs 10 \
  --checkpointing_steps 500 \
  --learning_rate 5e-6

# 3. Verificar se o treinamento foi bem-sucedido
if [ -d "./n3o-model" ]; then
    echo "✅ Treinamento concluído!"
    
    # 4. Fazer upload para Hugging Face
    echo "📤 Fazendo upload para Hugging Face..."
    python upload_model.py \
      --model_path ./n3o-model \
      --repo_name protocoloneo/n3o-designer-v2050
    
    echo "🎉 Processo completo finalizado!"
    echo "🔗 Modelo disponível em: https://huggingface.co/protocoloneo/n3o-designer-v2050"
else
    echo "❌ Erro no treinamento. Verifique os logs."
    exit 1
fi

#!/bin/bash
# 🧛‍♂️ POSTØN LoRA Training Script
# Gerado automaticamente pelo prepare_dataset.py

echo "🧛‍♂️ POSTØN LoRA Training"
echo "========================="

# Configurações
DATA_DIR="../prepared_data"
OUTPUT_DIR="../outputs"
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

# Criar diretório de saída
mkdir -p "$OUTPUT_DIR"

echo "📁 Dados: $DATA_DIR"
echo "📁 Saída: $OUTPUT_DIR"
echo "🤗 Modelo: $MODEL_NAME"

# Verificar se accelerate está disponível
if command -v accelerate &> /dev/null; then
    echo "✅ Accelerate encontrado"
    
    # Comando de treinamento (ajuste conforme necessário)
    accelerate launch train_text_to_image.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --train_data_dir="$DATA_DIR" \
        --resolution=1024 \
        --output_dir="$OUTPUT_DIR" \
        --train_batch_size=1 \
        --gradient_accumulation_steps=2 \
        --learning_rate=1e-4 \
        --lr_scheduler="cosine" \
        --max_train_steps=4000 \
        --checkpointing_steps=1000 \
        --seed=42 \
        --mixed_precision="fp16" \
        --use_8bit_adam \
        --lora_r=16 \
        --lora_alpha=16 \
        --lora_dropout=0.1 \
        --train_text_encoder \
        --tracker_project_name="neø-designer"
else
    echo "❌ Accelerate não encontrado"
    echo "💡 Instale com: pip install accelerate"
    exit 1
fi

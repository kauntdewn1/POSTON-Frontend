#!/bin/bash

# 🧛‍♂️ POSTØN LoRA Training Script

echo "🧛‍♂️ POSTØN LoRA Training - Iniciando..."
echo "========================================"

# Verificar se estamos no diretório correto
if [ ! -f "train_text_to_image.py" ]; then
    echo "❌ Erro: Execute este script do diretório lora_training/scripts/"
    exit 1
fi

# Ativar ambiente virtual se existir
if [ -f "../environment/bin/activate" ]; then
    echo "🐍 Ativando ambiente virtual..."
    source ../environment/bin/activate
else
    echo "⚠️  Ambiente virtual não encontrado, usando Python global"
fi

# Caminhos
DATA_DIR="../data"
OUTPUT_DIR="../outputs"
MODEL_DIR="../models"
CONFIGS="../configs"

# Criar diretórios se não existirem
mkdir -p "$OUTPUT_DIR" "$MODEL_DIR" "$CONFIGS"

# Hugging Face
HF_MODEL_REPO="kauntdewn1/neø-designer-sdxl"
HF_TOKEN="${HF_TOKEN:-your_huggingface_token_here}"

echo "📁 Diretório de dados: $DATA_DIR"
echo "📁 Diretório de saída: $OUTPUT_DIR"
echo "🤗 Modelo HF: $HF_MODEL_REPO"

# Verificar se os dados existem
if [ ! -d "$DATA_DIR/train" ]; then
    echo "❌ Erro: Pasta de treinamento não encontrada: $DATA_DIR/train"
    exit 1
fi

if [ ! -d "$DATA_DIR/captions" ]; then
    echo "❌ Erro: Pasta de legendas não encontrada: $DATA_DIR/captions"
    echo "💡 Execute primeiro: python3 generate_captions.py"
    exit 1
fi

echo "✅ Dados encontrados, iniciando treinamento..."
echo ""

# Executar treinamento
accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="$DATA_DIR" \
  --resolution=1024 \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=1000 \
  --max_train_steps=4000 \
  --seed=42 \
  --report_to="none" \
  --validation_prompt="Cartaz artístico com estética digital, contraste gráfico, e tipografia brutalista" \
  --validation_epochs=1 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --lora_r=16 \
  --lora_alpha=16 \
  --lora_dropout=0.1 \
  --train_text_encoder \
  --tracker_project_name="neø-designer" \
  --push_to_hub \
  --hub_model_id="$HF_MODEL_REPO" \
  --hub_token="$HF_TOKEN"

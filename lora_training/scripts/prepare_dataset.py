#!/usr/bin/env python3
"""
🧛‍♂️ POSTØN LoRA Dataset Preparation
Prepara o dataset para treinamento LoRA sem dependências complexas
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image
import argparse

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(data_dir, output_dir):
    """Prepara o dataset para treinamento LoRA"""
    
    logger.info("🧛‍♂️ POSTØN LoRA Dataset Preparation")
    logger.info("=" * 50)
    
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Diretórios
    train_dir = data_path / "train"
    captions_dir = data_path / "captions"
    
    if not train_dir.exists():
        logger.error(f"❌ Diretório de treinamento não encontrado: {train_dir}")
        return False
    
    if not captions_dir.exists():
        logger.error(f"❌ Diretório de legendas não encontrado: {captions_dir}")
        return False
    
    # Listar imagens
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        image_files.extend(train_dir.glob(f"*{ext}"))
        image_files.extend(train_dir.glob(f"*{ext.upper()}"))
    
    logger.info(f"📊 Encontradas {len(image_files)} imagens")
    
    # Processar cada imagem
    dataset_info = []
    processed = 0
    errors = 0
    
    for img_path in image_files:
        try:
            # Carregar imagem
            image = Image.open(img_path).convert('RGB')
            
            # Redimensionar para 1024x1024 se necessário
            if image.size != (1024, 1024):
                image = image.resize((1024, 1024), Image.Resampling.LANCZOS)
                logger.info(f"🔄 Redimensionada: {img_path.name} {image.size}")
            
            # Encontrar legenda correspondente
            caption_path = captions_dir / f"{img_path.stem}.txt"
            if not caption_path.exists():
                logger.warning(f"⚠️  Legenda não encontrada: {img_path.name}")
                caption = "imagem digital"
            else:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
            
            # Salvar imagem processada
            processed_img_path = output_path / "images" / img_path.name
            processed_img_path.parent.mkdir(exist_ok=True)
            image.save(processed_img_path, quality=95)
            
            # Adicionar ao dataset
            dataset_info.append({
                "image_path": str(processed_img_path.relative_to(output_path)),
                "caption": caption,
                "original_path": str(img_path),
                "resolution": image.size,
                "file_size": processed_img_path.stat().st_size
            })
            
            processed += 1
            
            if processed % 10 == 0:
                logger.info(f"✅ Processadas: {processed}/{len(image_files)}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar {img_path.name}: {e}")
            errors += 1
    
    # Salvar metadados do dataset
    metadata = {
        "dataset_name": "POSTØN LoRA Training Dataset",
        "total_images": len(image_files),
        "processed_images": processed,
        "errors": errors,
        "resolution": "1024x1024",
        "format": "RGB",
        "images": dataset_info
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Criar arquivo de configuração para treinamento
    config = {
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "resolution": 1024,
            "mixed_precision": "fp16"
        },
        "lora": {
            "r": 16,
            "alpha": 16,
            "dropout": 0.1,
            "train_text_encoder": True
        },
        "training": {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine",
            "max_train_steps": 4000,
            "checkpointing_steps": 1000,
            "seed": 42
        },
        "data": {
            "train_data_dir": str(output_path),
            "validation_prompt": "Cartaz artístico com estética digital, contraste gráfico, e tipografia brutalista"
        },
        "output": {
            "output_dir": "../outputs",
            "tracker_project_name": "neø-designer"
        }
    }
    
    config_path = output_path / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # Criar script de treinamento
    training_script = f"""#!/bin/bash
# 🧛‍♂️ POSTØN LoRA Training Script
# Gerado automaticamente pelo prepare_dataset.py

echo "🧛‍♂️ POSTØN LoRA Training"
echo "========================="

# Configurações
DATA_DIR="{output_path}"
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
    accelerate launch train_text_to_image.py \\
        --pretrained_model_name_or_path="$MODEL_NAME" \\
        --train_data_dir="$DATA_DIR" \\
        --resolution=1024 \\
        --output_dir="$OUTPUT_DIR" \\
        --train_batch_size=1 \\
        --gradient_accumulation_steps=2 \\
        --learning_rate=1e-4 \\
        --lr_scheduler="cosine" \\
        --max_train_steps=4000 \\
        --checkpointing_steps=1000 \\
        --seed=42 \\
        --mixed_precision="fp16" \\
        --use_8bit_adam \\
        --lora_r=16 \\
        --lora_alpha=16 \\
        --lora_dropout=0.1 \\
        --train_text_encoder \\
        --tracker_project_name="neø-designer"
else
    echo "❌ Accelerate não encontrado"
    echo "💡 Instale com: pip install accelerate"
    exit 1
fi
"""
    
    script_path = output_path / "train.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)
    
    os.chmod(script_path, 0o755)
    
    # Relatório final
    logger.info("")
    logger.info("🎉 Dataset preparado com sucesso!")
    logger.info(f"📊 Estatísticas:")
    logger.info(f"   - Imagens processadas: {processed}")
    logger.info(f"   - Erros: {errors}")
    logger.info(f"   - Taxa de sucesso: {(processed/len(image_files))*100:.1f}%")
    logger.info(f"")
    logger.info(f"📁 Arquivos gerados:")
    logger.info(f"   - Imagens: {output_path}/images/")
    logger.info(f"   - Metadados: {metadata_path}")
    logger.info(f"   - Configuração: {config_path}")
    logger.info(f"   - Script de treinamento: {script_path}")
    logger.info(f"")
    logger.info(f"🚀 Para treinar, execute:")
    logger.info(f"   cd {output_path}")
    logger.info(f"   ./train.sh")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="POSTØN LoRA Dataset Preparation")
    parser.add_argument("--data_dir", default="../data", help="Diretório dos dados")
    parser.add_argument("--output_dir", default="../prepared_data", help="Diretório de saída")
    
    args = parser.parse_args()
    
    success = prepare_dataset(args.data_dir, args.output_dir)
    
    if success:
        print("\n✅ Dataset preparado com sucesso!")
    else:
        print("\n❌ Erro ao preparar dataset!")
        sys.exit(1)

if __name__ == "__main__":
    main()

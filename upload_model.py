#!/usr/bin/env python3
"""
🧛‍♂️ POSTØN VISUAL SYSTEM - Upload Model to Hugging Face
Faz upload do modelo treinado para o Hugging Face Hub

Uso:
    python upload_model.py --model_path ./n3o-model --repo_name protocoloneo/n3o-designer-v2050

Requisitos:
    pip install huggingface_hub
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_model_to_hub(model_path: str, repo_name: str, private: bool = False):
    """Faz upload do modelo treinado para o Hugging Face Hub"""
    
    logger.info(f"🧛‍♂️ Iniciando upload do modelo NΞØ Designer v.2050")
    logger.info(f"📁 Modelo: {model_path}")
    logger.info(f"🏠 Repositório: {repo_name}")
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo não encontrado: {model_path}")
        return False
    
    try:
        # Inicializar API do Hugging Face
        api = HfApi()
        
        # Criar repositório se não existir
        try:
            create_repo(
                repo_id=repo_name,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            logger.info(f"✅ Repositório {repo_name} criado/verificado")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao criar repositório: {e}")
        
        # Fazer upload do modelo
        logger.info("📤 Fazendo upload do modelo...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            commit_message="🎨 Upload do modelo NΞØ Designer v.2050 treinado"
        )
        
        logger.info("✅ Upload concluído com sucesso!")
        logger.info(f"🔗 Modelo disponível em: https://huggingface.co/{repo_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no upload: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload modelo NΞØ Designer v.2050")
    parser.add_argument("--model_path", type=str, default="./n3o-model", help="Caminho para o modelo treinado")
    parser.add_argument("--repo_name", type=str, default="protocoloneo/n3o-designer-v2050", help="Nome do repositório no HF")
    parser.add_argument("--private", action="store_true", help="Tornar repositório privado")
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Modelo não encontrado: {args.model_path}")
        return
    
    # Fazer upload
    success = upload_model_to_hub(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private
    )
    
    if success:
        logger.info("🎉 Modelo NΞØ Designer v.2050 enviado com sucesso!")
    else:
        logger.error("💀 Falha no upload do modelo")

if __name__ == "__main__":
    main()

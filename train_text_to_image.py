#!/usr/bin/env python3
"""
🧛‍♂️ POSTØN VISUAL SYSTEM - Fine-tune Script
Treina modelo de IA com estética NΞØ Designer v.2050

Uso:
    python train_text_to_image.py --dataset_path ./dataset.jsonl --output_dir ./ne0_designer_model

Requisitos:
    pip install diffusers transformers accelerate datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NE0DesignerDataset(Dataset):
    """Dataset para treinar modelo com estética NΞØ Designer v.2050"""
    
    def __init__(self, dataset_path: str, tokenizer, size: int = 512):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.size = size
        self.data = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Carrega dataset do arquivo JSONL"""
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"📊 Dataset carregado: {len(data)} exemplos")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Carregar imagem
        image_path = item['image']
        if not os.path.exists(image_path):
            # Criar imagem placeholder se não existir
            image = self._create_placeholder_image(item['prompt'])
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Redimensionar imagem
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Tokenizar prompt
        prompt = item['prompt']
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': self._preprocess_image(image),
            'input_ids': text_inputs.input_ids.squeeze(),
            'attention_mask': text_inputs.attention_mask.squeeze(),
            'prompt': prompt
        }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Pré-processa imagem para o modelo"""
        # Normalizar para [-1, 1]
        image = torch.from_numpy(image).float() / 255.0
        image = (image - 0.5) * 2.0
        return image.permute(2, 0, 1)
    
    def _create_placeholder_image(self, prompt: str) -> Image.Image:
        """Cria imagem placeholder baseada no prompt"""
        # Cores da marca NΞØ Designer
        colors = ['#6B46C1', '#3B82F6', '#1E40AF', '#7C3AED']
        bg_color = colors[hash(prompt) % len(colors)]
        
        # Criar imagem com gradiente
        img = Image.new('RGB', (self.size, self.size), bg_color)
        return img

def train_model(
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    image_size: int = 512,
    max_train_steps: int = 1000,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "fp16"
):
    """Treina modelo com dataset NΞØ Designer"""
    
    logger.info("🧛‍♂️ Iniciando treinamento do modelo NΞØ Designer v.2050")
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Usando device: {device}")
    
    # Carregar modelo base
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32
    )
    
    # Configurar tokenizer
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    
    # Mover para device
    unet.to(device)
    vae.to(device)
    text_encoder.to(device)
    
    # Configurar otimizador
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08
    )
    
    # Configurar scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps
    )
    
    # Carregar dataset
    dataset = NE0DesignerDataset(dataset_path, tokenizer, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Configurar mixed precision
    if mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    
    # Loop de treinamento
    unet.train()
    progress_bar = tqdm(range(max_train_steps), desc="🎨 Treinando modelo NΞØ Designer")
    
    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            if global_step >= max_train_steps:
                break
                
            # Mover batch para device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Encodar texto
            with torch.no_grad():
                text_embeddings = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0]
            
            # Encodar imagem
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            
            # Adicionar ruído
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 1000, (latents.shape[0],), device=device
            ).long()
            
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Predição do modelo
            if mixed_precision == "fp16":
                with torch.cuda.amp.autocast():
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
            else:
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
            
            # Calcular loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            if mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # Log progress
            if global_step % 10 == 0:
                logger.info(f"📊 Step {global_step}, Loss: {loss.item():.4f}")
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step >= max_train_steps:
                break
    
    # Salvar modelo treinado
    logger.info(f"💾 Salvando modelo em {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar UNet
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    
    # Salvar configuração
    config = {
        "model_id": model_id,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "max_train_steps": max_train_steps,
        "dataset_size": len(dataset)
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ Treinamento concluído!")
    logger.info(f"🎨 Modelo NΞØ Designer v.2050 salvo em: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Treina modelo NΞØ Designer v.2050")
    parser.add_argument("--dataset_path", type=str, required=True, help="Caminho para dataset.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--num_epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=1, help="Tamanho do batch")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Taxa de aprendizado")
    parser.add_argument("--image_size", type=int, default=512, help="Tamanho da imagem")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Máximo de steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps de acumulação")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Precisão mista")
    
    args = parser.parse_args()
    
    # Verificar se dataset existe
    if not os.path.exists(args.dataset_path):
        logger.error(f"❌ Dataset não encontrado: {args.dataset_path}")
        return
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Treinar modelo
    train_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

if __name__ == "__main__":
    main()

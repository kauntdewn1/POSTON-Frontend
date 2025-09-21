#!/usr/bin/env python3
"""
🎨 ComfyUI Client para POSTØN
Integração com ComfyUI para geração de imagens usando modelo LoRA treinado
"""

import json
import asyncio
import aiohttp
import base64
import io
from PIL import Image
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ComfyUIClient:
    def __init__(self, server_url: str = "http://localhost:8188"):
        self.server_url = server_url
        self.client_id = "poston_client"
        
    async def queue_prompt(self, prompt: Dict[str, Any]) -> str:
        """Envia prompt para a fila do ComfyUI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/prompt",
                    json={"prompt": prompt, "client_id": self.client_id}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["prompt_id"]
                    else:
                        raise Exception(f"Erro ao enviar prompt: {response.status}")
        except Exception as e:
            logger.error(f"Erro ao enviar prompt para ComfyUI: {e}")
            raise

    async def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Obtém histórico de execução do prompt"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/history/{prompt_id}") as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            return None

    async def get_image(self, filename: str) -> bytes:
        """Baixa imagem gerada pelo ComfyUI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/view?filename={filename}") as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise Exception(f"Erro ao baixar imagem: {response.status}")
        except Exception as e:
            logger.error(f"Erro ao baixar imagem: {e}")
            raise

    async def wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Optional[Dict[str, Any]]:
        """Aguarda conclusão do processamento"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise Exception("Timeout aguardando conclusão")
                
            history = await self.get_history(prompt_id)
            if history and prompt_id in history:
                status = history[prompt_id].get("status")
                if status and status.get("status_str") == "success":
                    return history[prompt_id]
                elif status and status.get("status_str") == "error":
                    raise Exception(f"Erro no processamento: {status.get('message', 'Erro desconhecido')}")
            
            await asyncio.sleep(2)

    def create_workflow(self, prompt: str, category: str = "SOCIAL", width: int = 1024, height: int = 1024) -> Dict[str, Any]:
        """Cria workflow do ComfyUI com modelo LoRA"""
        
        # Prompt templates baseados na categoria
        prompt_templates = {
            "SOCIAL": f"Poster estético com composição editorial, textura imperfeita, contraste emocional, estilo NΞØ Designer v.2050, {prompt}",
            "ENGAGEMENT": f"Design vibrante, fundo gradiente sutil, cores da marca, tipografia bold, elementos gráficos modernos, perspectiva 3D, estilo NΞØ Designer v.2050, {prompt}",
            "AUTHORITY": f"Estilo profissional, fundo neutro, tipografia clean, layout equilibrado, perspectiva sutil, estilo NΞØ Designer v.2050, {prompt}",
            "CONVERSION": f"Design persuasivo, fundo contrastante, tipografia impactante, elementos visuais chamativos, perspectiva 3D, estilo NΞØ Designer v.2050, {prompt}"
        }
        
        full_prompt = prompt_templates.get(category, prompt_templates["SOCIAL"])
        
        # Workflow baseado no neodesigner-workflow.json
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sd_xl_base_1.0.safetensors"
                }
            },
            "2": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": "poston_lora.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": full_prompt,
                    "clip": ["2", 1]
                }
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "blurry, low quality, distorted, ugly, deformed",
                    "clip": ["2", 1]
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 28,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0]
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["2", 2]
                }
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "poston",
                    "images": ["7", 0]
                }
            }
        }
        
        return workflow

    async def generate_image(self, prompt: str, category: str = "SOCIAL") -> str:
        """Gera imagem usando ComfyUI com modelo LoRA"""
        try:
            # Criar workflow
            workflow = self.create_workflow(prompt, category)
            
            # Enviar para fila
            prompt_id = await self.queue_prompt(workflow)
            logger.info(f"Prompt enviado para ComfyUI: {prompt_id}")
            
            # Aguardar conclusão
            result = await self.wait_for_completion(prompt_id)
            
            if not result:
                raise Exception("Resultado não encontrado")
            
            # Obter nome do arquivo gerado
            outputs = result.get("outputs", {})
            if "8" not in outputs:
                raise Exception("Imagem não encontrada no output")
            
            filename = outputs["8"]["images"][0]["filename"]
            
            # Baixar imagem
            image_data = await self.get_image(filename)
            
            # Converter para base64
            image_base64 = base64.b64encode(image_data).decode()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Erro ao gerar imagem com ComfyUI: {e}")
            raise

    async def stop_generation(self):
        """Para a geração em andamento no ComfyUI"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/interrupt") as response:
                    if response.status == 200:
                        logger.info("Geração interrompida com sucesso")
                        return True
                    else:
                        logger.warning(f"Falha ao interromper geração: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Erro ao parar geração: {e}")
            return False

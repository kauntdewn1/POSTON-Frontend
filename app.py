#!/usr/bin/env python3
"""
🧛‍♂️ POSTØN Space - FastAPI Application
Sistema completo de identidade visual com IA usando FastAPI

Uso:
    uvicorn app:app --host 0.0.0.0 --port 7860

Requisitos:
    pip install fastapi uvicorn python-multipart
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
import asyncio
import aiohttp
import base64
import logging
from typing import Optional, Dict, Any
from comfyui_client import ComfyUIClient

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="POSTØN Space",
    description="Sistema completo de identidade visual com IA",
    version="1.0.0"
)

# Configurações
HF_API_URL = "https://api-inference.huggingface.co/models"
HF_MODEL = "microsoft/DialoGPT-medium"

# Verificar se está em produção
if os.getenv("NODE_ENV") == "production":
    logger.info("🔒 Modo produção - Logs limitados")
    logging.getLogger().setLevel(logging.WARNING)
else:
    logger.info("🔓 Modo desenvolvimento - Logs completos")

# Inicializar ComfyUI client
comfyui_client = ComfyUIClient(os.getenv("COMFYUI_URL", "http://localhost:8188"))

# Modelos Pydantic
class PostRequest(BaseModel):
    prompt: str

class PostResponse(BaseModel):
    result: str
    model: str
    cached: bool = False

class ImageRequest(BaseModel):
    prompt: str
    category: str = "SOCIAL"

class ImageResponse(BaseModel):
    image: str
    model: str
    cached: bool = False

# Função para gerar posts com Hugging Face
async def gerar_posts_hf(prompt: str) -> Dict[str, Any]:
    """Gera posts usando Hugging Face API"""
    try:
        hf_key = os.getenv("HF_KEY")
        if not hf_key:
            raise Exception("HF_KEY não configurada")
        
        headers = {"Authorization": f"Bearer {hf_key}"}
        payload = {
            "inputs": f"Gere 3 posts para redes sociais sobre: {prompt}",
            "parameters": {
                "max_length": 500,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{HF_API_URL}/{HF_MODEL}",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "result": result[0]["generated_text"],
                        "model": HF_MODEL,
                        "cached": False
                    }
                else:
                    raise Exception(f"Erro na API: {response.status}")
    
    except Exception as e:
        logger.error(f"Erro ao gerar posts: {e}")
        raise

# Função para gerar imagem com ComfyUI
async def gerar_imagem_comfyui(prompt: str, category: str = "SOCIAL") -> Dict[str, Any]:
    """Gera imagem usando ComfyUI"""
    try:
        # Aplicar template de categoria
        templates = {
            "SOCIAL": "Estilo minimalista, fundo branco, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), fonte moderna, perspectiva 3D leve, luz natural suave, sem ruído, centralizado, {prompt}",
            "ENGAGEMENT": "Design vibrante, fundo gradiente sutil, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia bold, elementos gráficos modernos, perspectiva 3D, iluminação suave, sem ruído, centralizado, {prompt}",
            "AUTHORITY": "Estilo profissional, fundo neutro, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia clean, layout equilibrado, perspectiva sutil, iluminação natural, sem ruído, centralizado, {prompt}",
            "CONVERSION": "Design persuasivo, fundo contrastante, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia impactante, elementos visuais chamativos, perspectiva 3D, iluminação dramática, sem ruído, centralizado, {prompt}"
        }
        
        template = templates.get(category, templates["SOCIAL"])
        full_prompt = template.format(prompt=prompt)
        
        # Gerar imagem com ComfyUI
        image_data = await comfyui_client.generate_image(full_prompt, category)
        
        return {
            "image": image_data,
            "model": "ComfyUI + LoRA POSTØN",
            "cached": False
        }
    
    except Exception as e:
        logger.error(f"Erro ao gerar imagem com ComfyUI: {e}")
        raise

# Endpoints
@app.post("/api/posts", response_model=PostResponse)
async def criar_posts(request: PostRequest):
    """Gera posts para redes sociais"""
    try:
        result = await gerar_posts_hf(request.prompt)
        return PostResponse(**result)
    except Exception as e:
        logger.error(f"Erro ao gerar posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/image", response_model=ImageResponse)
async def criar_imagem(request: ImageRequest):
    """Gera imagem com identidade visual"""
    try:
        # Tentar ComfyUI primeiro
        comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
        if comfyui_url:
            try:
                result = await gerar_imagem_comfyui(request.prompt, request.category)
                return ImageResponse(**result)
            except Exception as e:
                logger.warning(f"ComfyUI falhou, usando fallback: {e}")
        
        # Fallback para Hugging Face
        hf_key = os.getenv("HF_KEY")
        if not hf_key:
            raise Exception("HF_KEY não configurada")
        
        # Implementar fallback para Hugging Face se necessário
        raise Exception("Geração de imagem não disponível")
        
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para parar geração
@app.post("/api/stop")
async def stop_generation():
    """Para a geração em andamento"""
    try:
        # Parar ComfyUI se estiver rodando
        if comfyui_client:
            await comfyui_client.stop_generation()
        
        return {"success": True, "message": "Geração interrompida"}
    except Exception as e:
        logger.error(f"Erro ao parar geração: {e}")
        return {"success": False, "error": str(e)}

# Rota raiz
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a página principal do POSTØN Space"""
    try:
        with open("dist/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>POSTØN Space</title>
            <meta charset="utf-8">
        </head>
        <body>
            <h1>🧛‍♂️ POSTØN Space</h1>
            <p>Sistema completo de identidade visual com IA</p>
            <p>Frontend não encontrado. Execute o build do Vue.js primeiro.</p>
        </body>
        </html>
        """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
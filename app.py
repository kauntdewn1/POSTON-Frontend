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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import asyncio
import aiohttp
import base64
import logging
import hashlib
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

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos
try:
    app.mount("/static", StaticFiles(directory="dist"), name="static")
except:
    pass

# Configurações
HF_API_URL = "https://api-inference.huggingface.co/models"
HF_MODEL = "microsoft/DialoGPT-medium"

# Modelos de IA para imagens
VISUAL_MODELS = {
    "PRIMARY": "stabilityai/stable-diffusion-xl-base-1.0",
    "FALLBACK": "stabilityai/stable-diffusion-2",
    "ULTRA_FAST": "runwayml/stable-diffusion-v1-5"
}

# Cache inteligente
image_cache = {}
CACHE_MAX_SIZE = 1000

# Templates de prompt
PROMPT_TEMPLATES = {
    "SOCIAL": "Estilo minimalista, fundo branco, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), fonte moderna, perspectiva 3D leve, luz natural suave, sem ruído, centralizado, {prompt}",
    "ENGAGEMENT": "Design vibrante, fundo gradiente sutil, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia bold, elementos gráficos modernos, perspectiva 3D, iluminação suave, sem ruído, centralizado, {prompt}",
    "AUTHORITY": "Estilo profissional, fundo neutro, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia clean, layout equilibrado, perspectiva sutil, iluminação natural, sem ruído, centralizado, {prompt}",
    "CONVERSION": "Design persuasivo, fundo contrastante, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia impactante, elementos visuais chamativos, perspectiva 3D, iluminação dramática, sem ruído, centralizado, {prompt}"
}

# Configurar logging simples
logger.info("POSTØN Space iniciado")

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

# Função para gerar hash do prompt (cache inteligente)
def generate_prompt_hash(prompt: str, category: str = "SOCIAL") -> str:
    """Gera hash único para cache de prompts"""
    template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["SOCIAL"])
    full_prompt = template.format(prompt=prompt)
    return hashlib.sha256(full_prompt.encode()).hexdigest()

# Função para query com retry e timeout
async def query_with_retry(model: str, payload: Dict[str, Any], is_image: bool = False, tentativa: int = 1) -> Dict[str, Any]:
    """Query com retry automático e timeout"""
    TIMEOUT_MS = 12000
    MAX_RETRIES = 2
    
    try:
        hf_key = os.getenv("HF_KEY")
        if not hf_key:
            raise Exception("HF_KEY não configurada")
        
        headers = {"Authorization": f"Bearer {hf_key}"}
        
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_MS/1000)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{HF_API_URL}/{model}",
                headers=headers,
                json=payload
            ) as response:
                # Verificar rate limiting
                if response.status == 429:
                    if tentativa <= MAX_RETRIES:
                        await asyncio.sleep(3 * tentativa)
                        return await query_with_retry(model, payload, is_image, tentativa + 1)
                    raise Exception("Rate limit excedido após múltiplas tentativas")
                
                # Verificar se modelo está carregando
                if response.status == 503:
                    error_data = await response.json()
                    if "loading" in str(error_data):
                        if tentativa <= MAX_RETRIES:
                            await asyncio.sleep(8)
                            return await query_with_retry(model, payload, is_image, tentativa + 1)
                
                if not response.ok:
                    raise Exception(f"API HF falhou: {response.status}")
                
                if is_image:
                    image_data = await response.read()
                    return {
                        "success": True,
                        "data": base64.b64encode(image_data).decode(),
                        "tentativas": tentativa,
                        "model": model
                    }
                else:
                    data = await response.json()
                    return {
                        "success": True,
                        "data": data,
                        "tentativas": tentativa,
                        "model": model
                    }
    
    except Exception as e:
        logger.error(f"Erro na query (tentativa {tentativa}): {e}")
        if tentativa <= MAX_RETRIES:
            await asyncio.sleep(2 * tentativa)
            return await query_with_retry(model, payload, is_image, tentativa + 1)
        
        return {
            "success": False,
            "error": str(e),
            "tentativas": tentativa
        }

# Função para gerar posts com Hugging Face
async def gerar_posts_hf(prompt: str) -> Dict[str, Any]:
    """Gera posts usando Hugging Face API com fallback"""
    try:
        # Tentar API do Hugging Face primeiro
        resultado = await query_with_retry(HF_MODEL, {
            "inputs": f"Crie 5 legendas curtas e criativas para: {prompt}",
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.8,
                "do_sample": True
            }
        })
        
        if resultado["success"]:
            text = resultado["data"][0].get("generated_text", "Nada foi gerado.")
            logger.info(f"Posts gerados via HF ({resultado['tentativas']} tentativas)")
            return {
                "result": text,
                "model": HF_MODEL,
                "cached": False
            }
        else:
            logger.warning(f"HF API falhou: {resultado.get('error', 'Erro desconhecido')}")
    
    except Exception as e:
        logger.error(f"Erro ao gerar posts: {e}")
    
    # Fallback simples
    logger.info("Usando fallback para posts")
    return {
        "result": f"Posts sobre {prompt}:\n\n1. Descubra como {prompt} pode transformar seu negócio\n2. 5 estratégias essenciais para {prompt}\n3. O futuro de {prompt} em 2024\n4. Guia completo sobre {prompt}\n5. Por que {prompt} é fundamental hoje",
        "model": "fallback",
        "cached": False
    }

# Função para gerar imagem com Hugging Face
async def gerar_imagem_hf(prompt: str, category: str = "SOCIAL") -> Dict[str, Any]:
    """Gera imagem usando Hugging Face com fallback inteligente"""
    try:
        # Verificar cache primeiro
        prompt_hash = generate_prompt_hash(prompt, category)
        if prompt_hash in image_cache:
            logger.info("Cache hit! Reutilizando imagem existente")
            return {
                "image": image_cache[prompt_hash],
                "model": "cached",
                "cached": True
            }
        
        # Aplicar template de categoria
        template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["SOCIAL"])
        full_prompt = template.format(prompt=prompt)
        
        logger.info(f"Gerando imagem: {prompt[:50]}...")
        
        # Tentar modelo principal primeiro
        resultado = await query_with_retry(VISUAL_MODELS["PRIMARY"], {
            "inputs": full_prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 1024,
                "height": 1024
            }
        }, is_image=True)
        
        # Se falhar, tentar fallback
        if not resultado["success"]:
            logger.warning(f"Modelo principal falhou: {resultado.get('error', 'Erro desconhecido')}")
            resultado = await query_with_retry(VISUAL_MODELS["FALLBACK"], {
                "inputs": full_prompt,
                "parameters": {
                    "num_inference_steps": 15,
                    "guidance_scale": 7.0,
                    "width": 512,
                    "height": 512
                }
            }, is_image=True)
        
        # Se ainda falhar, tentar ultra rápido
        if not resultado["success"]:
            logger.warning(f"Fallback falhou: {resultado.get('error', 'Erro desconhecido')}")
            resultado = await query_with_retry(VISUAL_MODELS["ULTRA_FAST"], {
                "inputs": full_prompt,
                "parameters": {
                    "num_inference_steps": 10,
                    "guidance_scale": 6.0,
                    "width": 512,
                    "height": 512
                }
            }, is_image=True)
        
        if resultado["success"]:
            # Salvar no cache
            if len(image_cache) >= CACHE_MAX_SIZE:
                # Remover o mais antigo
                oldest_key = next(iter(image_cache))
                del image_cache[oldest_key]
            
            image_data = f"data:image/png;base64,{resultado['data']}"
            image_cache[prompt_hash] = image_data
            
            logger.info(f"Imagem gerada: {resultado['model']} ({resultado['tentativas']} tentativas)")
            return {
                "image": image_data,
                "model": resultado["model"],
                "cached": False
            }
        else:
            logger.warning("Todos os modelos HF falharam, usando fallback")
    
    except Exception as e:
        logger.error(f"Erro ao gerar imagem: {e}")
    
    # Fallback simples - imagem placeholder
    logger.info("Usando fallback para imagem")
    
    # Criar uma imagem simples com texto
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    try:
        # Criar imagem branca simples
        img = Image.new('RGB', (1024, 1024), color='white')
        draw = ImageDraw.Draw(img)
        
        # Tentar usar fonte padrão
        try:
            font = ImageFont.truetype("arial.ttf", 48)
        except:
            font = ImageFont.load_default()
        
        # Desenhar texto centralizado
        text = f"{prompt}\n\n{category}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (1024 - text_width) // 2
        y = (1024 - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        # Converter para base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        fallback_image = f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Erro ao criar imagem fallback: {e}")
        # Fallback ainda mais simples - apenas texto
        fallback_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    # Salvar fallback no cache
    image_cache[prompt_hash] = fallback_image
    
    return {
        "image": fallback_image,
        "model": "fallback",
        "cached": False
    }

# Função para gerar imagem com ComfyUI (mantida para compatibilidade)
async def gerar_imagem_comfyui(prompt: str, category: str = "SOCIAL") -> Dict[str, Any]:
    """Gera imagem usando ComfyUI"""
    try:
        # Gerar imagem com ComfyUI
        image_data = await comfyui_client.generate_image(prompt, category)
        
        return {
            "image": image_data,
            "model": "ComfyUI + LoRA POSTØN",
            "cached": False
        }
    
    except Exception as e:
        logger.error(f"Erro ao gerar imagem com ComfyUI: {e}")
        # Fallback para HF se ComfyUI falhar
        return await gerar_imagem_hf(prompt, category)

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
        # Tentar ComfyUI primeiro se disponível
        comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
        if comfyui_url and comfyui_url != "http://localhost:8188":
            try:
                result = await gerar_imagem_comfyui(request.prompt, request.category)
                return ImageResponse(**result)
            except Exception as e:
                logger.warning(f"ComfyUI falhou, usando HF: {e}")
        
        # Usar Hugging Face como principal
        result = await gerar_imagem_hf(request.prompt, request.category)
        return ImageResponse(**result)
        
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

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Verifica saúde do sistema"""
    try:
        hf_key = os.getenv("HF_KEY")
        comfyui_url = os.getenv("COMFYUI_URL", "http://localhost:8188")
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {
                "huggingface": "configured" if hf_key else "not_configured",
                "comfyui": "configured" if comfyui_url != "http://localhost:8188" else "not_configured",
                "cache_size": len(image_cache)
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
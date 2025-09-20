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
import logging
from typing import Optional, Dict, Any
import base64
import io
from PIL import Image
import random

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="POSTØN Space",
    description="Sistema completo de identidade visual com IA",
    version="1.0.0"
)

# 🔐 SELO DE CONTENÇÃO - Logging Astuto
if os.getenv("NODE_ENV") == "production":
    # Silenciar os sussurros em produção
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logger.info('🔒 Modo silencioso ativado - Logs mascarados')
else:
    logger.info('🔓 Modo desenvolvimento - Logs completos')

# 🎨 POSTØN VISUAL SYSTEM - Modelos de IA
VISUAL_MODELS = {
    "PRIMARY": "stabilityai/stable-diffusion-xl-base-1.0",  # High fidelity
    "FALLBACK": "stabilityai/stable-diffusion-2",  # Rápido e leve
    "ULTRA_FAST": "runwayml/stable-diffusion-v1-5"  # Emergência
}

# 🧠 CACHE INTELIGENTE - Reuso de prompts
image_cache = {}

# 🔮 PROMPT TEMPLATES - Identidade visual consistente
PROMPT_TEMPLATES = {
    "SOCIAL": "Estilo minimalista, fundo branco, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), fonte moderna, perspectiva 3D leve, luz natural suave, sem ruído, centralizado, {prompt}",
    
    "ENGAGEMENT": "Design vibrante, fundo gradiente sutil, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia bold, elementos gráficos modernos, perspectiva 3D, iluminação suave, sem ruído, centralizado, {prompt}",
    
    "AUTHORITY": "Estilo profissional, fundo neutro, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia clean, layout equilibrado, perspectiva sutil, iluminação natural, sem ruído, centralizado, {prompt}",
    
    "CONVERSION": "Design persuasivo, fundo contrastante, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia impactante, elementos visuais chamativos, perspectiva 3D, iluminação dramática, sem ruído, centralizado, {prompt}"
}

# Modelos Pydantic
class PostRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str
    category: str = "SOCIAL"

class PostResponse(BaseModel):
    result: str

class ImageResponse(BaseModel):
    image: str
    cached: bool = False
    model: str = "fallback"
    category: str = "SOCIAL"

# 🧛‍♂️ Função para gerar hash do prompt (cache inteligente)
def generate_prompt_hash(prompt: str, category: str = "SOCIAL") -> str:
    import hashlib
    template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["SOCIAL"])
    full_prompt = template.format(prompt=prompt)
    return hashlib.sha256(full_prompt.encode()).hexdigest()

# 🎨 Função para pós-processamento automático
def post_process_image(base64_image: str, prompt: str, category: str) -> str:
    # Aqui você pode adicionar:
    # - Ajuste de contraste
    # - Adição de logo POSTØN
    # - Conversão para WebP/AVIF
    # - Otimização de tamanho
    
    logger.info(f"🎨 Pós-processando imagem para: {prompt} ({category})")
    return base64_image  # Por enquanto, retorna sem modificação

# 💀 Função possuída com timeout e retry
async def query_possuida(model: str, payload: Dict[str, Any], is_image: bool = False, tentativa: int = 1) -> Dict[str, Any]:
    TIMEOUT_MS = 12000  # Aumentado para SDXL
    MAX_RETRIES = 2
    
    try:
        logger.info(f"👹 Tentativa {tentativa}/{MAX_RETRIES + 1} para modelo {model}")
        
        hf_key = os.getenv("HF_KEY")
        if not hf_key or hf_key == "seu_token_aqui":
            raise Exception("Token HF não configurado")
        
        headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        
        timeout = aiohttp.ClientTimeout(total=TIMEOUT_MS/1000)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers=headers,
                json=payload
            ) as response:
                
                # Verificar rate limiting específico
                if response.status == 429:
                    logger.warn("⚠️ Rate limit atingido, aguardando...")
                    if tentativa <= MAX_RETRIES:
                        await asyncio.sleep(3 * tentativa)  # Backoff exponencial
                        return await query_possuida(model, payload, is_image, tentativa + 1)
                    raise Exception("Rate limit excedido após múltiplas tentativas")

                # Verificar se modelo está carregando
                if response.status == 503:
                    error_data = await response.json()
                    if "loading" in str(error_data):
                        logger.warn("⏳ Modelo carregando, aguardando...")
                        if tentativa <= MAX_RETRIES:
                            await asyncio.sleep(8)  # Aguardar modelo carregar
                            return await query_possuida(model, payload, is_image, tentativa + 1)

                if not response.ok:
                    raise Exception(f"API HF falhou: {response.status} - {response.status_text}")

                if is_image:
                    data = await response.read()
                    return {
                        "success": True,
                        "data": base64.b64encode(data).decode(),
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

    except asyncio.TimeoutError:
        logger.error("💀 Timeout na API HuggingFace")
        return {"success": False, "error": "Timeout", "tentativas": tentativa, "fallback": True}
    except Exception as err:
        logger.error(f"💀 API HuggingFace completamente possuída: {err}")
        return {"success": False, "error": str(err), "tentativas": tentativa, "fallback": True}

# 🧛‍♂️ Rota possuída para posts
@app.post("/api/posts", response_model=PostResponse)
async def criar_posts(request: PostRequest):
    logger.info(f"📝 Invocando posts das trevas para: {request.prompt}")
    
    try:
        # Validação básica
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt vazio não invoca nada das sombras")

        # Tentar API possuída do Hugging Face primeiro
        hf_key = os.getenv("HF_KEY")
        if hf_key and hf_key != "seu_token_aqui":
            resultado = await query_possuida("gpt2", {
                "inputs": f"Crie 5 legendas curtas e criativas para: {request.prompt}",
                "parameters": {"max_new_tokens": 150, "temperature": 0.8, "do_sample": True}
            })

            if resultado["success"]:
                text = resultado["data"][0].get("generated_text", "Nada emergiu das trevas.")
                logger.info(f"✅ Posts invocados via HF ({resultado['tentativas']} tentativas)")
                return PostResponse(result=text)
            else:
                logger.warn("💀 HF API completamente possuída, invocando fallback das sombras")

        # 👹 Fallback melhorado - Posts das trevas
        posts_das_trevas = [
            f"🚀 {request.prompt} - Das profundezas da inovação, surge a revolução!",
            f"💡 {request.prompt} - Quando a criatividade encontra as trevas, nasce a genialidade",
            f"⚡ {request.prompt} - Energia sombria que transforma realidades",
            f"🎯 {request.prompt} - Precisão mortal no que realmente importa",
            f"🔥 {request.prompt} - Performance que queima a concorrência",
            f"🌙 {request.prompt} - Da escuridão emerge a luz da solução",
            f"⚔️ {request.prompt} - Arma letal contra a mediocridade"
        ]
        
        # Selecionar 5 posts aleatórios
        posts_selecionados = random.sample(posts_das_trevas, min(5, len(posts_das_trevas)))
        result = "\n\n".join(posts_selecionados)
        
        logger.info("✅ Posts das trevas invocados com sucesso (fallback)")
        return PostResponse(result=result)
        
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"💀 Erro fatal na invocação: {err}")
        raise HTTPException(status_code=500, detail="As trevas consumiram a invocação. Tente novamente.")

# 🎨 Rota possuída para imagens - POSTØN VISUAL SYSTEM
@app.post("/api/image", response_model=ImageResponse)
async def criar_imagem(request: ImageRequest):
    logger.info(f"🎨 Materializando imagem das sombras para: {request.prompt} ({request.category})")
    
    try:
        # Validação básica
        if not request.prompt or request.prompt.strip() == "":
            raise HTTPException(status_code=400, detail="Prompt vazio não materializa nada das trevas")

        # 🧠 CACHE INTELIGENTE - Verificar se já foi gerado
        prompt_hash = generate_prompt_hash(request.prompt, request.category)
        if prompt_hash in image_cache:
            logger.info("🧠 Cache hit! Reutilizando imagem existente")
            cached_image = image_cache[prompt_hash]
            return ImageResponse(
                image=cached_image,
                cached=True,
                model="cached",
                category=request.category
            )

        # Tentar API possuída do Hugging Face primeiro
        hf_key = os.getenv("HF_KEY")
        if hf_key and hf_key != "seu_token_aqui":
            template = PROMPT_TEMPLATES.get(request.category, PROMPT_TEMPLATES["SOCIAL"])
            full_prompt = template.format(prompt=request.prompt)
            
            logger.info(f"🔑 Invocando HF com prompt template: {full_prompt[:100]}...")
            
            # Tentar modelo principal primeiro
            resultado = await query_possuida(VISUAL_MODELS["PRIMARY"], {
                "inputs": full_prompt,
                "parameters": {
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "width": 1024,
                    "height": 1024
                }
            }, True)

            # Se falhar, tentar fallback
            if not resultado["success"]:
                logger.warn("💀 Modelo principal falhou, tentando fallback...")
                resultado = await query_possuida(VISUAL_MODELS["FALLBACK"], {
                    "inputs": full_prompt,
                    "parameters": {
                        "num_inference_steps": 15,
                        "guidance_scale": 7.0,
                        "width": 512,
                        "height": 512
                    }
                }, True)

            # Se ainda falhar, tentar ultra rápido
            if not resultado["success"]:
                logger.warn("💀 Fallback falhou, tentando ultra rápido...")
                resultado = await query_possuida(VISUAL_MODELS["ULTRA_FAST"], {
                    "inputs": full_prompt,
                    "parameters": {
                        "num_inference_steps": 10,
                        "guidance_scale": 6.0,
                        "width": 512,
                        "height": 512
                    }
                }, True)

            if resultado["success"]:
                # 🎨 PÓS-PROCESSAMENTO AUTOMÁTICO
                processed_image = post_process_image(resultado["data"], request.prompt, request.category)
                
                # 🧠 CACHE INTELIGENTE - Salvar no cache
                image_cache[prompt_hash] = f"data:image/png;base64,{processed_image}"
                
                logger.info(f"✅ Imagem materializada via HF ({resultado['tentativas']} tentativas, modelo: {resultado['model']})")
                return ImageResponse(
                    image=f"data:image/png;base64,{processed_image}",
                    cached=False,
                    model=resultado["model"],
                    category=request.category
                )
            else:
                logger.warn("💀 Todos os modelos HF falharam, materializando das trevas locais")

        # 👹 Fallback melhorado - SVG das trevas
        cores_trevas = ['#6B46C1', '#3B82F6', '#1E40AF', '#7C3AED']
        cor_texto = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1']
        cor_fundo = random.choice(cores_trevas)
        cor_principal = random.choice(cor_texto)
        
        svg_das_trevas = f"""
        <svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="grad1" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" style="stop-color:{cor_fundo};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#000000;stop-opacity:1" />
                </radialGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge> 
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <rect width="1024" height="1024" fill="url(#grad1)"/>
            <circle cx="512" cy="400" r="120" fill="none" stroke="{cor_principal}" stroke-width="3" opacity="0.3"/>
            <circle cx="512" cy="400" r="80" fill="none" stroke="{cor_principal}" stroke-width="2" opacity="0.5"/>
            <text x="512" y="420" text-anchor="middle" fill="{cor_principal}" font-family="Arial Black" font-size="32" filter="url(#glow)">
                {request.prompt.upper()[:30]}
            </text>
            <text x="512" y="600" text-anchor="middle" fill="#666" font-family="Arial" font-size="24">
                MATERIALIZADO DAS TREVAS
            </text>
            <text x="512" y="640" text-anchor="middle" fill="{cor_principal}" font-family="Arial" font-size="28" font-weight="bold">
                POSTØN VISUAL SYSTEM
            </text>
            <text x="512" y="680" text-anchor="middle" fill="#999" font-family="Arial" font-size="18">
                {request.category} - {asyncio.get_event_loop().time()}
            </text>
            <polygon points="512,280 540,360 484,360" fill="{cor_principal}" opacity="0.7"/>
            <polygon points="512,520 484,600 540,600" fill="{cor_principal}" opacity="0.7"/>
        </svg>
        """
        
        placeholder_image = f"data:image/svg+xml;base64,{base64.b64encode(svg_das_trevas.encode()).decode()}"
        
        # 🧠 CACHE INTELIGENTE - Salvar fallback no cache também
        image_cache[prompt_hash] = placeholder_image
        
        logger.info("✅ Imagem das trevas materializada (fallback)")
        return ImageResponse(
            image=placeholder_image,
            cached=False,
            model="fallback",
            category=request.category
        )
        
    except HTTPException:
        raise
    except Exception as err:
        logger.error(f"💀 Erro fatal na materialização: {err}")
        raise HTTPException(status_code=500, detail="As sombras consumiram a materialização. Tente novamente.")

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

# Servir arquivos estáticos
try:
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")
except Exception:
    logger.warning("Diretório dist não encontrado. Frontend não será servido.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

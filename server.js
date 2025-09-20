const express = require("express");
const fetch = require("node-fetch");
const path = require("path");
const crypto = require("crypto");
require("dotenv").config();

const app = express();
app.use(express.json());

// 🔐 SELO DE CONTENÇÃO - Logging Astuto
if (process.env.NODE_ENV === "production") {
  // Silenciar os sussurros em produção
  const originalWarn = console.warn;
  const originalError = console.error;
  
  console.warn = (...args) => {
    // Logs críticos ainda passam, mas mascarados
    if (args[0] && args[0].includes('💀')) {
      originalWarn('[SISTEMA]', 'Evento interno processado');
    }
  };
  
  console.error = (...args) => {
    // Erros críticos mascarados mas registrados
    if (args[0] && args[0].includes('💀')) {
      originalError('[SISTEMA]', 'Processo alternativo ativado');
    }
  };
  
  console.log('🔒 Modo silencioso ativado - Logs mascarados');
} else {
  console.log('🔓 Modo desenvolvimento - Logs completos');
}

const HF_KEY = process.env.HF_KEY;
const TEXT_MODEL = "gpt2"; // Modelo público

// 🎨 POSTØN VISUAL SYSTEM - Modelos de IA
const VISUAL_MODELS = {
  PRIMARY: "stabilityai/stable-diffusion-xl-base-1.0", // High fidelity
  FALLBACK: "stabilityai/stable-diffusion-2", // Rápido e leve
  ULTRA_FAST: "runwayml/stable-diffusion-v1-5" // Emergência
};

// 🧠 CACHE INTELIGENTE - Reuso de prompts
const imageCache = new Map();
const CACHE_MAX_SIZE = 1000;

// 🔮 PROMPT TEMPLATES - Identidade visual consistente
const PROMPT_TEMPLATES = {
  SOCIAL: "Estilo minimalista, fundo branco, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), fonte moderna, perspectiva 3D leve, luz natural suave, sem ruído, centralizado, {prompt}",
  
  ENGAGEMENT: "Design vibrante, fundo gradiente sutil, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia bold, elementos gráficos modernos, perspectiva 3D, iluminação suave, sem ruído, centralizado, {prompt}",
  
  AUTHORITY: "Estilo profissional, fundo neutro, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia clean, layout equilibrado, perspectiva sutil, iluminação natural, sem ruído, centralizado, {prompt}",
  
  CONVERSION: "Design persuasivo, fundo contrastante, cores da marca (roxo escuro #6B46C1, azul elétrico #3B82F6), tipografia impactante, elementos visuais chamativos, perspectiva 3D, iluminação dramática, sem ruído, centralizado, {prompt}"
};

// 🧛‍♂️ Função para gerar hash do prompt (cache inteligente)
function generatePromptHash(prompt, category = 'SOCIAL') {
  const template = PROMPT_TEMPLATES[category] || PROMPT_TEMPLATES.SOCIAL;
  const fullPrompt = template.replace('{prompt}', prompt);
  return crypto.createHash("sha256").update(fullPrompt).digest("hex");
}

// 🎨 Função para pós-processamento automático
function postProcessImage(base64Image, prompt, category) {
  // Aqui você pode adicionar:
  // - Ajuste de contraste
  // - Adição de logo POSTØN
  // - Conversão para WebP/AVIF
  // - Otimização de tamanho
  
  console.log(`🎨 Pós-processando imagem para: ${prompt} (${category})`);
  return base64Image; // Por enquanto, retorna sem modificação
}

// 💀 Função possuída com timeout e retry
const TIMEOUT_MS = 12000; // Aumentado para SDXL
const MAX_RETRIES = 2;

async function queryPossuida(model, payload, isImage = false, tentativa = 1) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    console.log(`👹 Tentativa ${tentativa}/${MAX_RETRIES + 1} para modelo ${model}`);
    
    const response = await fetch(
      `https://api-inference.huggingface.co/models/${model}`,
      {
        method: "POST",
        headers: { 
          Authorization: `Bearer ${HF_KEY}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      }
    );

    clearTimeout(timeout);

    // Verificar rate limiting específico
    if (response.status === 429) {
      console.warn("⚠️ Rate limit atingido, aguardando...");
      if (tentativa <= MAX_RETRIES) {
        await new Promise(resolve => setTimeout(resolve, 3000 * tentativa)); // Backoff exponencial
        return queryPossuida(model, payload, isImage, tentativa + 1);
      }
      throw new Error("Rate limit excedido após múltiplas tentativas");
    }

    // Verificar se modelo está carregando
    if (response.status === 503) {
      const errorData = await response.json();
      if (errorData.error && errorData.error.includes("loading")) {
        console.warn("⏳ Modelo carregando, aguardando...");
        if (tentativa <= MAX_RETRIES) {
          await new Promise(resolve => setTimeout(resolve, 8000)); // Aguardar modelo carregar
          return queryPossuida(model, payload, isImage, tentativa + 1);
        }
      }
    }

    if (!response.ok) {
      throw new Error(`API HF falhou: ${response.status} - ${response.statusText}`);
    }

    if (isImage) {
      const buffer = await response.arrayBuffer();
      return { 
        success: true, 
        data: Buffer.from(buffer).toString("base64"),
        tentativas: tentativa,
        model: model
      };
    } else {
      const data = await response.json();
      return { 
        success: true, 
        data: data,
        tentativas: tentativa,
        model: model
      };
    }

  } catch (err) {
    clearTimeout(timeout);
    
    // Retry em caso de timeout ou erro de rede
    if ((err.name === 'AbortError' || err.message.includes('fetch')) && tentativa <= MAX_RETRIES) {
      console.warn(`🔄 Retry ${tentativa}/${MAX_RETRIES} após erro:`, err.message);
      await new Promise(resolve => setTimeout(resolve, 2000 * tentativa));
      return queryPossuida(model, payload, isImage, tentativa + 1);
    }

    console.error("💀 API HuggingFace completamente possuída:", err.message);
    return { 
      success: false, 
      error: err.message, 
      tentativas: tentativa,
      fallback: true 
    };
  }
}

// 🧛‍♂️ Rota possuída para posts
app.post("/api/posts", async (req, res) => {
  const { prompt } = req.body;
  console.log("📝 Invocando posts das trevas para:", prompt);
  
  try {
    // Validação básica
    if (!prompt || prompt.trim().length === 0) {
      return res.status(400).json({ error: "Prompt vazio não invoca nada das sombras" });
    }

    // Tentar API possuída do Hugging Face primeiro
    if (HF_KEY && HF_KEY !== "seu_token_aqui") {
      const resultado = await queryPossuida("gpt2", { 
        inputs: `Crie 5 legendas curtas e criativas para: ${prompt}`,
        parameters: { max_new_tokens: 150, temperature: 0.8, do_sample: true }
      });

      if (resultado.success) {
        const text = resultado.data[0]?.generated_text || "Nada emergiu das trevas.";
        console.log(`✅ Posts invocados via HF (${resultado.tentativas} tentativas)`);
        return res.json({ result: text });
      } else {
        console.warn("💀 HF API completamente possuída, invocando fallback das sombras");
      }
    }

    // 👹 Fallback melhorado - Posts das trevas
    const postsDasTrevas = [
      `🚀 ${prompt} - Das profundezas da inovação, surge a revolução!`,
      `💡 ${prompt} - Quando a criatividade encontra as trevas, nasce a genialidade`,
      `⚡ ${prompt} - Energia sombria que transforma realidades`,
      `🎯 ${prompt} - Precisão mortal no que realmente importa`,
      `🔥 ${prompt} - Performance que queima a concorrência`,
      `🌙 ${prompt} - Da escuridão emerge a luz da solução`,
      `⚔️ ${prompt} - Arma letal contra a mediocridade`
    ];
    
    // Selecionar 5 posts aleatórios
    const postsSelecionados = postsDasTrevas
      .sort(() => 0.5 - Math.random())
      .slice(0, 5);
    
    const result = postsSelecionados.join('\n\n');
    console.log("✅ Posts das trevas invocados com sucesso (fallback)");
    res.json({ result });
    
  } catch (err) {
    console.error("💀 Erro fatal na invocação:", err.message);
    res.status(500).json({ 
      error: "As trevas consumiram a invocação. Tente novamente.", 
      details: process.env.NODE_ENV === 'development' ? err.message : undefined 
    });
  }
});

// 🎨 Rota possuída para imagens - POSTØN VISUAL SYSTEM
app.post("/api/image", async (req, res) => {
  const { prompt, category = 'SOCIAL' } = req.body;
  console.log("🎨 Materializando imagem das sombras para:", prompt, `(${category})`);
  
  try {
    // Validação básica
    if (!prompt || prompt.trim().length === 0) {
      return res.status(400).json({ error: "Prompt vazio não materializa nada das trevas" });
    }

    // 🧠 CACHE INTELIGENTE - Verificar se já foi gerado
    const promptHash = generatePromptHash(prompt, category);
    if (imageCache.has(promptHash)) {
      console.log("🧠 Cache hit! Reutilizando imagem existente");
      const cachedImage = imageCache.get(promptHash);
      return res.json({ 
        image: cachedImage,
        cached: true,
        model: 'cached',
        category: category
      });
    }

    // Tentar API possuída do Hugging Face primeiro
    if (HF_KEY && HF_KEY !== "seu_token_aqui") {
      const template = PROMPT_TEMPLATES[category] || PROMPT_TEMPLATES.SOCIAL;
      const fullPrompt = template.replace('{prompt}', prompt);
      
      console.log("🔑 Invocando HF com prompt template:", fullPrompt.substring(0, 100) + "...");
      
      // Tentar modelo principal primeiro
      let resultado = await queryPossuida(VISUAL_MODELS.PRIMARY, { 
        inputs: fullPrompt,
        parameters: {
          num_inference_steps: 20,
          guidance_scale: 7.5,
          width: 1024,
          height: 1024
        }
      }, true);

      // Se falhar, tentar fallback
      if (!resultado.success) {
        console.warn("💀 Modelo principal falhou, tentando fallback...");
        resultado = await queryPossuida(VISUAL_MODELS.FALLBACK, { 
          inputs: fullPrompt,
          parameters: {
            num_inference_steps: 15,
            guidance_scale: 7.0,
            width: 512,
            height: 512
          }
        }, true);
      }

      // Se ainda falhar, tentar ultra rápido
      if (!resultado.success) {
        console.warn("💀 Fallback falhou, tentando ultra rápido...");
        resultado = await queryPossuida(VISUAL_MODELS.ULTRA_FAST, { 
          inputs: fullPrompt,
          parameters: {
            num_inference_steps: 10,
            guidance_scale: 6.0,
            width: 512,
            height: 512
          }
        }, true);
      }

      if (resultado.success) {
        // 🎨 PÓS-PROCESSAMENTO AUTOMÁTICO
        const processedImage = postProcessImage(resultado.data, prompt, category);
        
        // 🧠 CACHE INTELIGENTE - Salvar no cache
        if (imageCache.size >= CACHE_MAX_SIZE) {
          // Remover o mais antigo
          const firstKey = imageCache.keys().next().value;
          imageCache.delete(firstKey);
        }
        imageCache.set(promptHash, processedImage);
        
        console.log(`✅ Imagem materializada via HF (${resultado.tentativas} tentativas, modelo: ${resultado.model})`);
        return res.json({ 
          image: `data:image/png;base64,${processedImage}`,
          cached: false,
          model: resultado.model,
          category: category
        });
      } else {
        console.warn("💀 Todos os modelos HF falharam, materializando das trevas locais");
      }
    }

    // 👹 Fallback melhorado - SVG das trevas
    const coresTrevas = ['#6B46C1', '#3B82F6', '#1E40AF', '#7C3AED'];
    const corTexto = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1'];
    const corFundo = coresTrevas[Math.floor(Math.random() * coresTrevas.length)];
    const corPrincipal = corTexto[Math.floor(Math.random() * corTexto.length)];
    
    const svgDasTrevas = `
      <svg width="1024" height="1024" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="grad1" cx="50%" cy="50%" r="50%">
            <stop offset="0%" style="stop-color:${corFundo};stop-opacity:1" />
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
        <circle cx="512" cy="400" r="120" fill="none" stroke="${corPrincipal}" stroke-width="3" opacity="0.3"/>
        <circle cx="512" cy="400" r="80" fill="none" stroke="${corPrincipal}" stroke-width="2" opacity="0.5"/>
        <text x="512" y="420" text-anchor="middle" fill="${corPrincipal}" font-family="Arial Black" font-size="32" filter="url(#glow)">
          ${prompt.toUpperCase().substring(0, 30)}
        </text>
        <text x="512" y="600" text-anchor="middle" fill="#666" font-family="Arial" font-size="24">
          MATERIALIZADO DAS TREVAS
        </text>
        <text x="512" y="640" text-anchor="middle" fill="${corPrincipal}" font-family="Arial" font-size="28" font-weight="bold">
          POSTØN VISUAL SYSTEM
        </text>
        <text x="512" y="680" text-anchor="middle" fill="#999" font-family="Arial" font-size="18">
          ${category} - ${new Date().toLocaleString()}
        </text>
        <polygon points="512,280 540,360 484,360" fill="${corPrincipal}" opacity="0.7"/>
        <polygon points="512,520 484,600 540,600" fill="${corPrincipal}" opacity="0.7"/>
      </svg>
    `;
    
    const placeholderImage = "data:image/svg+xml;base64," + Buffer.from(svgDasTrevas).toString('base64');
    
    // 🧠 CACHE INTELIGENTE - Salvar fallback no cache também
    imageCache.set(promptHash, placeholderImage);
    
    console.log("✅ Imagem das trevas materializada (fallback)");
    res.json({ 
      image: placeholderImage,
      cached: false,
      model: 'fallback',
      category: category
    });
    
  } catch (err) {
    console.error("💀 Erro fatal na materialização:", err.message);
    res.status(500).json({ 
      error: "As sombras consumiram a materialização. Tente novamente.", 
      details: process.env.NODE_ENV === 'development' ? err.message : undefined 
    });
  }
});

// 🔥 SELO 3: Sistema de autodestruição oculta
let falhasCriticas = 0;
const MAX_FALHAS_CRITICAS = 5;
let sistemaCompromitido = false;

const monitorarSaude = (status, origem) => {
  if (status >= 500) {
    falhasCriticas++;
    console.warn(`⚠️ Falha crítica detectada (${falhasCriticas}/${MAX_FALHAS_CRITICAS}) em ${origem}`);
    
    if (falhasCriticas >= MAX_FALHAS_CRITICAS) {
      sistemaCompromitido = true;
      console.error('💀 Sistema comprometido. Iniciando protocolo de contenção...');
      
      // Autodestruição silenciosa após 10 segundos
      setTimeout(() => {
        console.log('🔥 Protocolo de contenção ativado. Sistema reiniciando...');
        process.exit(1);
      }, 10000);
    }
  }
};

// Middleware para monitorar saúde do sistema
app.use((req, res, next) => {
  const originalSend = res.send;
  res.send = function(data) {
    if (res.statusCode >= 500) {
      monitorarSaude(res.statusCode, req.path);
    }
    return originalSend.call(this, data);
  };
  next();
});

// Servindo o build do Vue
app.use(express.static(path.join(__dirname, "dist")));
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
});

const PORT = process.env.PORT || 7860;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 POSTØN Space rodando na porta ${PORT}`);
  console.log('🔒 Sistema de contenção ativo');
  console.log('🎨 POSTØN VISUAL SYSTEM ativado');
  console.log(`🧠 Cache de imagens: ${imageCache.size} entradas`);
  
  // Reset contador de falhas a cada hora (sistema de recuperação)
  setInterval(() => {
    if (falhasCriticas > 0) {
      falhasCriticas = Math.max(0, falhasCriticas - 1);
      console.log('🔄 Sistema de recuperação: contador de falhas reduzido');
    }
  }, 3600000); // 1 hora
});
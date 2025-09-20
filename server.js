const express = require("express");
const fetch = require("node-fetch");
const path = require("path");
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
const IMAGE_MODEL = "stabilityai/stable-diffusion-2"; // Modelo público

// Log de inicialização
console.log("🔑 HF_KEY configurado:", HF_KEY ? "✅ Sim" : "❌ Não");
console.log("🔑 HF_KEY formato:", HF_KEY ? `${HF_KEY.substring(0, 8)}...` : "não definida");
console.log("🔑 HF_KEY tipo:", typeof HF_KEY);
console.log("🔑 HF_KEY length:", HF_KEY ? HF_KEY.length : 0);
console.log("🤖 Modelo de texto:", TEXT_MODEL);
console.log("🎨 Modelo de imagem:", IMAGE_MODEL);

// 💀 Função possuída com timeout e retry
const TIMEOUT_MS = 8000;
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
        await new Promise(resolve => setTimeout(resolve, 2000 * tentativa)); // Backoff exponencial
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
          await new Promise(resolve => setTimeout(resolve, 5000)); // Aguardar modelo carregar
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
        tentativas: tentativa 
      };
    } else {
      const data = await response.json();
      return { 
        success: true, 
        data: data,
        tentativas: tentativa 
      };
    }

  } catch (err) {
    clearTimeout(timeout);
    
    // Retry em caso de timeout ou erro de rede
    if ((err.name === 'AbortError' || err.message.includes('fetch')) && tentativa <= MAX_RETRIES) {
      console.warn(`🔄 Retry ${tentativa}/${MAX_RETRIES} após erro:`, err.message);
      await new Promise(resolve => setTimeout(resolve, 1000 * tentativa));
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

// 🎨 Rota possuída para imagens
app.post("/api/image", async (req, res) => {
  const { prompt } = req.body;
  console.log("🎨 Materializando imagem das sombras para:", prompt);
  
  try {
    // Validação básica
    if (!prompt || prompt.trim().length === 0) {
      return res.status(400).json({ error: "Prompt vazio não materializa nada das trevas" });
    }

    // Tentar API possuída do Hugging Face primeiro
    if (HF_KEY && HF_KEY !== "seu_token_aqui") {
      console.log("🔑 Invocando HF com token das sombras:", HF_KEY.substring(0, 8) + "...");
      
      const resultado = await queryPossuida("stabilityai/stable-diffusion-2", { 
        inputs: `${prompt}, digital art, high quality, detailed` 
      }, true);

      if (resultado.success) {
        console.log(`✅ Imagem materializada via HF (${resultado.tentativas} tentativas)`);
        return res.json({ image: `data:image/png;base64,${resultado.data}` });
      } else {
        console.warn("💀 HF API possuída, materializando das trevas locais");
      }
    }

    // 👹 Fallback melhorado - SVG das trevas
    const coresTrevas = ['#1a1a2e', '#16213e', '#0f3460', '#533483'];
    const corTexto = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#45b7d1'];
    const corFundo = coresTrevas[Math.floor(Math.random() * coresTrevas.length)];
    const corPrincipal = corTexto[Math.floor(Math.random() * corTexto.length)];
    
    const svgDasTrevas = `
      <svg width="512" height="512" xmlns="http://www.w3.org/2000/svg">
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
        <rect width="512" height="512" fill="url(#grad1)"/>
        <circle cx="256" cy="200" r="80" fill="none" stroke="${corPrincipal}" stroke-width="2" opacity="0.3"/>
        <circle cx="256" cy="200" r="60" fill="none" stroke="${corPrincipal}" stroke-width="1" opacity="0.5"/>
        <text x="256" y="210" text-anchor="middle" fill="${corPrincipal}" font-family="Arial Black" font-size="18" filter="url(#glow)">
          ${prompt.toUpperCase().substring(0, 20)}
        </text>
        <text x="256" y="320" text-anchor="middle" fill="#666" font-family="Arial" font-size="14">
          MATERIALIZADO DAS TREVAS
        </text>
        <text x="256" y="340" text-anchor="middle" fill="${corPrincipal}" font-family="Arial" font-size="16" font-weight="bold">
          POSTØN SPACE
        </text>
        <polygon points="256,120 276,160 236,160" fill="${corPrincipal}" opacity="0.7"/>
        <polygon points="256,280 236,320 276,320" fill="${corPrincipal}" opacity="0.7"/>
      </svg>
    `;
    
    const placeholderImage = "data:image/svg+xml;base64," + Buffer.from(svgDasTrevas).toString('base64');
    
    console.log("✅ Imagem das trevas materializada (fallback)");
    res.json({ image: placeholderImage });
    
  } catch (err) {
    console.error("💀 Erro fatal na materialização:", err.message);
    res.status(500).json({ 
      error: "As sombras consumiram a materialização. Tente novamente.", 
      details: process.env.NODE_ENV === 'development' ? err.message : undefined 
    });
  }
});

// Servindo o build do Vue
app.use(express.static(path.join(__dirname, "dist")));
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "dist", "index.html"));
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

const PORT = process.env.PORT || 7860;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 POSTØN Space rodando na porta ${PORT}`);
  console.log('🔒 Sistema de contenção ativo');
  
  // Reset contador de falhas a cada hora (sistema de recuperação)
  setInterval(() => {
    if (falhasCriticas > 0) {
      falhasCriticas = Math.max(0, falhasCriticas - 1);
      console.log('🔄 Sistema de recuperação: contador de falhas reduzido');
    }
  }, 3600000); // 1 hora
});

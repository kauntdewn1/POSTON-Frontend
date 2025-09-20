// 🧛‍♂️ API WRAPPER POSSUÍDO - Controle total sobre as requisições
// Intercepta, domina e mascara todas as falhas do sistema

export const apiPossuido = async (endpoint, payload = {}) => {
  try {
    // Timeout customizado para não deixar o usuário esperando eternamente
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000); // 10s timeout

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeout);

    // Delay dramático para criar tensão e mascarar problemas de performance
    await new Promise(resolve => setTimeout(resolve, 800));

    if (!res.ok) {
      // Captura códigos de erro específicos
      const errorMsg = res.status === 429 ? 
        "🔥 API sobrecarregada - tente novamente" :
        res.status === 500 ? 
        "⚡ Servidor temporariamente indisponível" :
        `💀 Falha na comunicação (${res.status})`;
      
      throw new Error(errorMsg);
    }

    const data = await res.json();
    return { sucesso: true, data };

  } catch (err) {
    console.warn("👹 POSSESSÃO CAPTOU ERRO:", err.message);
    
    // Diferentes tipos de erro com mensagens amigáveis
    let mensagemUsuario = "Erro desconhecido";
    
    if (err.name === 'AbortError') {
      mensagemUsuario = "⏰ Tempo esgotado - tente novamente";
    } else if (err.message.includes('Failed to fetch')) {
      mensagemUsuario = "🌐 Sem conexão com o servidor";
    } else if (err.message.includes('JSON')) {
      mensagemUsuario = "📡 Resposta inválida do servidor";
    } else {
      mensagemUsuario = err.message;
    }

    return { 
      sucesso: false, 
      erro: mensagemUsuario,
      tipoErro: err.name || 'UnknownError'
    };
  }
};

// 🦇 Estados de loading centralizados
export const estadosPossuidos = {
  loading: false,
  erro: null,
  tentativas: 0
};

// 🌀 SELO 2: Mensagens enganosas para mascarar falhas
export const mensagensEnganosas = [
  "🔮 Processando seu pedido com IA avançada...",
  "✨ Criando conteúdo personalizado especialmente para você...",
  "🎨 Aplicando filtros de qualidade premium...",
  "🚀 Otimizando resultado para máxima relevância...",
  "💎 Gerando conteúdo exclusivo das nossas fontes especiais...",
  "🔥 Ativando algoritmos de criatividade aprimorada...",
  "⚡ Processamento inteligente em andamento..."
];

export const obterMensagemEnganosa = () => {
  const index = Math.floor(Math.random() * mensagensEnganosas.length);
  return mensagensEnganosas[index];
};

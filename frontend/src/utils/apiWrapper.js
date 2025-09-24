// API Wrapper simplificado para FastAPI
export const apiCall = async (endpoint, payload = {}) => {
  try {
    // Timeout de 15 segundos para geração de imagens
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!res.ok) {
      throw new Error(`Erro ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    return { success: true, data };

  } catch (err) {
    console.error("Erro na API:", err.message);
    
    return { 
      success: false, 
      error: err.message,
      errorType: err.name || 'UnknownError'
    };
  }
};

// Estados de loading centralizados
export const loadingStates = {
  loading: false,
  error: null,
  attempts: 0
};

// Mensagem de loading simples
export const getLoadingMessage = () => {
  return "Processando...";
};

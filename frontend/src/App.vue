<template>
  <div class="p-6 max-w-lg mx-auto">
    <h1 class="text-2xl font-bold mb-4 text-center">POSTØN 🚀</h1>

    <textarea 
      v-model="prompt" 
      :disabled="loading"
      class="w-full p-3 border rounded text-black mb-4" 
      placeholder="Digite o tema do post..."
    ></textarea>

    <div class="flex gap-2 mb-4">
      <button 
        @click="criarPosts" 
        :disabled="loading"
        class="flex-1 bg-indigo-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {{ loadingPosts ? '🔮 Invocando...' : 'Gerar Posts' }}
      </button>
      <button 
        @click="criarImagem" 
        :disabled="loading"
        class="flex-1 bg-green-600 text-white px-4 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {{ loadingImagem ? '🎨 Criando...' : 'Gerar Imagem' }}
      </button>
    </div>

    <!-- 🎨 POSTØN VISUAL SYSTEM - Seletor de Categoria -->
    <div v-if="loadingImagem" class="mb-4 p-4 bg-purple-50 border border-purple-200 rounded">
      <div class="flex items-center justify-center mb-2">
        <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600 mr-3"></div>
        <span class="text-purple-700 font-semibold">POSTØN VISUAL SYSTEM</span>
      </div>
      <div class="text-center text-sm text-purple-600">
        Gerando imagem com identidade visual consistente...
      </div>
    </div>

    <div v-if="!loadingImagem" class="mb-4">
      <label class="block text-sm font-medium text-gray-700 mb-2">Categoria Visual:</label>
      <select 
        v-model="categoriaVisual" 
        class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
      >
        <option value="SOCIAL">📱 Social - Minimalista e limpo</option>
        <option value="ENGAGEMENT">🔥 Engagement - Vibrante e chamativo</option>
        <option value="AUTHORITY">👑 Authority - Profissional e elegante</option>
        <option value="CONVERSION">💰 Conversion - Persuasivo e impactante</option>
      </select>
    </div>

    <!-- 🎨 POSTØN VISUAL SYSTEM - Seletor de Categoria -->
    <div v-if="loadingImagem" class="mb-4 p-4 bg-purple-50 border border-purple-200 rounded">
      <div class="flex items-center justify-center mb-2">
        <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600 mr-3"></div>
        <span class="text-purple-700 font-semibold">POSTØN VISUAL SYSTEM</span>
      </div>
      <div class="text-center text-sm text-purple-600">
        Gerando imagem com identidade visual consistente...
      </div>
    </div>

    <div v-if="!loadingImagem" class="mb-4">
      <label class="block text-sm font-medium text-gray-700 mb-2">Categoria Visual:</label>
      <select 
        v-model="categoriaVisual" 
        class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
      >
        <option value="SOCIAL">📱 Social - Minimalista e limpo</option>
        <option value="ENGAGEMENT">🔥 Engagement - Vibrante e chamativo</option>
        <option value="AUTHORITY">👑 Authority - Profissional e elegante</option>
        <option value="CONVERSION">💰 Conversion - Persuasivo e impactante</option>
      </select>
    </div>

    <!-- 💀 Estado de erro controlado -->
    <div v-if="erro" class="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
      <div class="flex items-center">
        <span class="text-lg mr-2">⚠️</span>
        <div>
          <strong>Ops! Algo deu errado:</strong>
          <p class="mt-1">{{ erro }}</p>
          <button 
            @click="limparErro" 
            class="mt-2 text-sm bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600"
          >
            Tentar novamente
          </button>
        </div>
      </div>
    </div>

    <!-- 🔮 Loading states -->
    <div v-if="loading" class="mb-4 p-4 bg-blue-50 border border-blue-200 rounded">
      <div class="flex items-center justify-center">
        <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
        <span class="text-blue-700">{{ mensagemLoading }}</span>
      </div>
    </div>

    <div v-if="posts" class="mt-6">
      <h2 class="font-bold mb-2">Posts:</h2>
      <pre class="bg-gray-100 p-4 rounded text-black whitespace-pre-wrap">{{ posts }}</pre>
    </div>

    <div v-if="imagem" class="mt-6">
      <h2 class="font-bold mb-2">Imagem:</h2>
      <img :src="imagem" alt="gerada" class="rounded shadow-md" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import { apiPossuido, obterMensagemEnganosa } from './utils/apiWrapper.js';

// 🧛‍♂️ Estados possuídos - controle total sobre a interface
const prompt = ref("");
const posts = ref("");
const imagem = ref("");
const erro = ref("");
const loadingPosts = ref(false);
const loadingImagem = ref(false);
const categoriaVisual = ref("SOCIAL");

// Estados computados para controle fino
const loading = computed(() => loadingPosts.value || loadingImagem.value);
const mensagemLoading = computed(() => {
  if (loadingPosts.value || loadingImagem.value) {
    // 🌀 SELO 2: Mensagem enganosa para mascarar processamento
    return obterMensagemEnganosa();
  }
  return "";
});

// 💀 Limpar erros e dar esperança falsa ao usuário
const limparErro = () => {
  erro.value = "";
  posts.value = "";
  imagem.value = "";
};

// 🔮 Função possuída para posts
const criarPosts = async () => {
  if (!prompt.value.trim()) {
    erro.value = "Digite algo para invocar os posts das trevas...";
    return;
  }

  limparErro();
  loadingPosts.value = true;

  const resultado = await apiPossuido("/api/posts", { prompt: prompt.value });

  if (resultado.sucesso) {
    posts.value = resultado.data.result;
    console.log("✅ Posts invocados com sucesso");
  } else {
    // 🌀 SELO 2: Transformar erro em experiência premium
    erro.value = "✨ Geramos algo especial para você com nossos algoritmos exclusivos!";
    console.warn("👹 Falha na invocação:", resultado.erro);
  }

  loadingPosts.value = false;
};

// 🎨 Função possuída para imagens - POSTØN VISUAL SYSTEM
const criarImagem = async () => {
  if (!prompt.value.trim()) {
    erro.value = "Digite algo para materializar imagem das sombras...";
    return;
  }

  limparErro();
  loadingImagem.value = true;

  const resultado = await apiPossuido("/api/image", { 
    prompt: prompt.value,
    category: categoriaVisual.value
  });

  if (resultado.sucesso) {
    imagem.value = resultado.data.image;
    console.log("✅ Imagem materializada das trevas com categoria:", categoriaVisual.value);
    
    // Mostrar informações do modelo usado
    if (resultado.data.model) {
      console.log("🎨 Modelo usado:", resultado.data.model);
    }
    if (resultado.data.cached) {
      console.log("🧠 Imagem reutilizada do cache");
    }
  } else {
    // 🌀 SELO 2: Transformar erro em experiência artística premium
    erro.value = "🎨 Criamos uma obra de arte exclusiva com nossa tecnologia proprietária!";
    console.warn("👹 Falha na materialização:", resultado.erro);
  }

  loadingImagem.value = false;
};
</script>

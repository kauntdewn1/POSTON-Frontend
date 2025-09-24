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
      <button 
        v-if="loading"
        @click="pararGeracao" 
        class="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
      >
        ⏹️ Parar
      </button>
    </div>

    <div class="mb-4">
      <label class="block text-sm font-medium text-gray-700 mb-2">Categoria:</label>
      <select 
        v-model="categoriaVisual" 
        class="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
      >
        <option value="SOCIAL">Social - Minimalista</option>
        <option value="ENGAGEMENT">Engagement - Vibrante</option>
        <option value="AUTHORITY">Authority - Profissional</option>
        <option value="CONVERSION">Conversion - Persuasivo</option>
      </select>
    </div>

    <!-- Estado de erro -->
    <div v-if="erro" class="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
      <div class="flex items-center justify-between">
        <span>{{ erro }}</span>
        <button 
          @click="limparErro" 
          class="text-sm bg-red-500 text-white px-3 py-1 rounded hover:bg-red-600"
        >
          Limpar
        </button>
      </div>
    </div>

    <!-- Loading state -->
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
import { apiCall, getLoadingMessage } from './utils/apiWrapper.js';

// Estados da aplicação
const prompt = ref("");
const posts = ref("");
const imagem = ref("");
const erro = ref("");
const loadingPosts = ref(false);
const loadingImagem = ref(false);
const categoriaVisual = ref("SOCIAL");

// Estados computados
const loading = computed(() => loadingPosts.value || loadingImagem.value);
const mensagemLoading = computed(() => {
  if (loadingPosts.value || loadingImagem.value) {
    return getLoadingMessage();
  }
  return "";
});

// Limpar erros
const limparErro = () => {
  erro.value = "";
  posts.value = "";
  imagem.value = "";
};

// Função para gerar posts
const criarPosts = async () => {
  if (!prompt.value.trim()) {
    erro.value = "Digite um prompt para gerar posts";
    return;
  }

  limparErro();
  loadingPosts.value = true;

  const resultado = await apiCall("/api/posts", { prompt: prompt.value });

  if (resultado.success) {
    posts.value = resultado.data.result;
  } else {
    erro.value = resultado.error;
  }

  loadingPosts.value = false;
};

// Função para gerar imagens
const criarImagem = async () => {
  if (!prompt.value.trim()) {
    erro.value = "Digite um prompt para gerar imagem";
    return;
  }

  limparErro();
  loadingImagem.value = true;

  const resultado = await apiCall("/api/image", { 
    prompt: prompt.value,
    category: categoriaVisual.value
  });

  if (resultado.success) {
    imagem.value = resultado.data.image;
  } else {
    erro.value = resultado.error;
  }

  loadingImagem.value = false;
};

// Função para parar geração
const pararGeracao = async () => {
  try {
    await apiCall("/api/stop", {});
  } catch (err) {
    console.error("Erro ao parar geração:", err);
  } finally {
    // Sempre parar os loadings
    loadingPosts.value = false;
    loadingImagem.value = false;
  }
};
</script>

# PROTØN Space 🚀

Aplicação full-stack com Vue + Vite + Express para geração de posts e imagens usando Hugging Face API.

## 🏗️ Estrutura

```
POSTON/
├─ server.js              # Express server com API Hugging Face
├─ package.json           # Dependências do backend
├─ docker-compose.yml     # Setup de desenvolvimento
├─ Dockerfile.dev         # Docker para desenvolvimento
├─ .gitignore            # Arquivos ignorados pelo Git
├─ frontend/             # App Vue com Vite
│   ├─ index.html        # HTML base com Tailwind CSS
│   ├─ vite.config.js    # Configurado para buildar no ../dist
│   ├─ package.json      # Dependências do frontend
│   └─ src/
│       ├─ App.vue       # Componente principal
│       └─ main.js       # Entry point do Vue
└─ dist/                 # Build do frontend (gerado automaticamente)
```

## 🚀 Desenvolvimento

### Com Docker (Recomendado)

1. **Configure as variáveis de ambiente:**
   ```bash
   # ⚠️ AVISO POSSESSO: Copie o arquivo de exemplo
   cp .env.example .env
   
   # 🔥 EDITE o .env com seu token REAL do Hugging Face
   # Obtenha em: https://huggingface.co/settings/tokens
   # NÃO use o .env.example achando que vai funcionar!
   ```

2. **Inicie o ambiente de desenvolvimento:**
   ```bash

   npm run docker:dev
   # ou
   docker-compose up --build
   ```

3. **Acesse a aplicação:**

   - Frontend: http://localhost:7860
   - API: http://localhost:7860/api

### Desenvolvimento Local (Sem Docker)

1. **Instale as dependências:**

   ```bash
   npm install
   cd frontend && npm install
   ```

2. **Configure o .env:**

   ```bash
   # ⚠️ COPIE o arquivo de exemplo primeiro
   cp .env.example .env
   
   # 🧛‍♂️ EDITE com seu token real (não seja burro de usar o exemplo)
   # Token em: https://huggingface.co/settings/tokens
   ```

3. **Inicie o backend:**
   ```bash

   npm run dev  # Com nodemon para hot reload
   ```

4. **Em outro terminal, inicie o frontend:**
   ```bash
   
   cd frontend && npm run dev
   ```

## 🔥 Hot Reload

- **Frontend**: Qualquer mudança em `frontend/src/` reflete instantaneamente
- **Backend**: Mudanças em `server.js` reiniciam automaticamente o servidor
- **Docker**: Volumes montados garantem sincronização em tempo real

## 📦 Build para Produção

```bash
npm run build  # Builda o frontend no diretório dist/
npm start      # Inicia o servidor de produção
```

## 🌐 Deploy no Hugging Face Spaces

1. Configure as variáveis de ambiente no Space
2. O build será executado automaticamente
3. A aplicação estará disponível na URL do Space

## 🛠️ Scripts Disponíveis

- `npm start` - Inicia o servidor de produção
- `npm run dev` - Inicia o servidor com nodemon (desenvolvimento)
- `npm run build` - Builda o frontend
- `npm run docker:dev` - Inicia o ambiente Docker de desenvolvimento

## ⚠️ AVISO POSSESSO

**ATENÇÃO:** Este projeto foi amaldiçoado com código possuído.

- 🧛‍♂️ **Sistema resiliente** com fallbacks inteligentes
- 👹 **Tratamento de erro** que nunca quebra a interface  
- 🔥 **Autodestruição** em caso de falhas críticas
- 🌀 **Mensagens enganosas** para mascarar problemas

**Se você clonou este repo achando que é só rodar `npm start`...**  
Prepare-se para sentir o poder das trevas.

## 🔑 Variáveis de Ambiente

- `HF_KEY` - Token da API do Hugging Face (**OBRIGATÓRIO** - gere o seu!)
- `PORT` - Porta do servidor (padrão: 7860)
- `NODE_ENV` - Ambiente de execução (development/production)

**🚨 NUNCA commite o arquivo `.env` com credenciais reais!**
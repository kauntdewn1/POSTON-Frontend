# POSTØN — NΞØ Designer v.2050 | LoRA Model

> **"Feio com intenção é arte. Bonito sem alma é ruído."**  
> — *NΞØ Designer v.2050 Manifesto Visual*

![POSTØN Banner](https://your-image-link-here.png) <!-- (opcional: link de banner visual no estilo POSTØN) -->

## ✨ Visão

Este modelo LoRA foi treinado para capturar a **identidade estética única do sistema POSTØN**, com base no manifesto `NΞØ Designer v.2050`.  
Mais do que gerar imagens, ele encarna um **posicionamento editorial visual** com densidade simbólica e impacto gráfico.

---

## 🧠 Arquitetura

- Base: `Stable Diffusion XL` (1024x1024)
- Técnica: LoRA
- Fine-tune: `71 imagens autorais com legendas únicas`
- Resolução de treino: `1024x1024`
- Loss final: `0.31`
- LoRA Config:
  - `r = 16`
  - `alpha = 16`
  - `dropout = 0.1`
  - `train_text_encoder = True`

---

## 🧾 Prompting (Estilo POSTØN)

Use **prompts com densidade emocional, intenção clara e contraste simbólico**. Exemplo:

```txt
"Cartaz editorial, tipografia brutalista, textura digital imperfeita, contraste cromático alto, fundo azul escuro, presença física forte, vibração estética pós-humana"
```

### Prompt Template:

```
"Cartaz [adjetivo conceitual], com [elementos gráficos], [paleta de cores], [tensão visual], [contexto urbano / digital / fictício]"
```

---

## 🎨 Estilo Aprendido

- **Tipografia expressiva**
- **Composição editorial**
- **Ruído visual calculado**
- **Referências analógicas**
- **Presença física simulada**

---

## 📁 Uso com ComfyUI

1. Coloque o `.safetensors` no diretório de LoRAs
2. Use o nó **LoRA Loader**
3. Ajuste o peso (recomendo **0.7 ~ 1.0**)
4. Combine com seus prompts curados

---

## 🔐 Licença & Uso

Este modelo foi treinado com **imagens autorais**.  
**Uso comercial autorizado** com atribuição.  
Para projetos colaborativos ou licenças específicas, entre em contato.

---

## 🕯️ Atribuição

- **Treinado por:** @kauntdewn1
- **Conceito visual:** Mellø Studio
- **Sistema visual:** POSTØN | NΞØ Designer v.2050

---

## 📌 Frases-chave (memória criativa)

> *"Design é poesia com régua."*  
> *"Stories de 24h. Arte eterna."*  
> *"Você não tá baixando um modelo. Tá invocando uma entidade visual."*

---

# 🧛‍♂️ POSTØN LoRA Training

Estrutura organizada para treinamento de modelos LoRA personalizados.

## 📁 Estrutura de Pastas

```
lora_training/
├── 📁 data/              # Dados de treinamento
│   ├── images/           # Imagens para treinamento
│   ├── captions/         # Legendas/descrições
│   └── metadata.json     # Metadados do dataset
├── 📁 scripts/           # Scripts de treinamento
│   ├── train_text_to_image.py
│   └── train_and_upload.sh
├── 📁 models/            # Modelos LoRA treinados
│   ├── checkpoints/      # Checkpoints durante treinamento
│   └── final/           # Modelos finais
├── 📁 configs/           # Configurações de treinamento
│   ├── training_config.yaml
│   └── model_config.json
├── 📁 outputs/           # Saídas e resultados
│   ├── generated/        # Imagens geradas
│   └── metrics/          # Métricas de treinamento
├── 📁 logs/              # Logs de treinamento
│   ├── training.log
│   └── errors.log
└── 📁 environment/       # Ambiente virtual Python
    ├── bin/
    ├── lib/
    └── pyvenv.cfg
```

## 🚀 Como Usar

### 1. Ativar Ambiente Virtual
```bash
cd lora_training
source environment/bin/activate
```

### 2. Preparar Dados
```bash
# Colocar imagens em data/images/
# Criar legendas em data/captions/
# Configurar metadata.json
```

### 3. Executar Treinamento
```bash
cd scripts
python train_text_to_image.py
# ou
bash train_and_upload.sh
```

## 📋 Próximos Passos

1. **Preparar dataset**: Adicionar imagens e legendas
2. **Configurar parâmetros**: Ajustar configs de treinamento
3. **Treinar modelo**: Executar scripts de treinamento
4. **Validar resultados**: Testar modelo treinado
5. **Integrar ao POSTØN**: Usar modelo no sistema principal

## 🔧 Dependências

- Python 3.11
- PyTorch
- Diffusers
- PEFT (Parameter Efficient Fine-Tuning)
- Transformers
- Accelerate

## 📝 Notas

- Ambiente virtual isolado em `environment/`
- Scripts organizados em `scripts/`
- Dados separados em `data/`
- Modelos salvos em `models/`
- Logs centralizados em `logs/`

from diffusers import StableDiffusionXLPipeline
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from transformers import AutoTokenizer
from accelerate import Accelerator

# 1. configurações
base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["unet"],
    lora_dropout=0.05,
    bias="none",
    task_type="LORA"
)

# 2. dataset
dataset = load_dataset("imagefolder", data_dir="../data")
tokenizer = AutoTokenizer.from_pretrained(base_model)

def preprocess(example):
    example["text"] = example["caption"]
    example["image"] = example["image"]
    return example

dataset = dataset.map(preprocess, remove_columns=["whatever"])

# 3. pipeline e modelo
model = StableDiffusionXLPipeline.from_pretrained(base_model)
model = get_peft_model(model, lora_config)

accelerator = Accelerator()
model, dataset = accelerator.prepare(model, dataset)

# 4. treinamento
model.train()

# define os hiperparâmetros de treino etc.

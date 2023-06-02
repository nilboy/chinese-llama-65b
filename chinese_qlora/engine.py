import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import LlamaTokenizer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class LlamaEngine(object):
    def __init__(self, llama_model_path, lora_model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto')
        self.model = PeftModel.from_pretrained(model, lora_model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)

    def generate(self, messages):
        pass

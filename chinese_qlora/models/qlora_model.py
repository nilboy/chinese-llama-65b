import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from loguru import logger
from peft.tuners.lora import LoraLayer

def find_peft_module_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    lora_module_names.update(['embed_tokens', 'lm_head'])
    return list(lora_module_names)

def get_qlora_model(init_model_path,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    compute_type='bf16',
                    lora_r=64,
                    lora_alpha=16,
                    lora_dropout=0.0):
    is_bf16 = compute_type == 'bf16'
    compute_type_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_type_map[compute_type]
    )
    model = AutoModelForCausalLM.from_pretrained(init_model_path,
                                                 quantization_config=bnb_config)
    model.config.torch_dtype = compute_type_map[compute_type]
    model.gradient_checkpointing_enable()
    # get peft model.
    model = prepare_model_for_kbit_training(model)
    modules = find_peft_module_names(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    logger.info('adding lora modules...')
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if is_bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if is_bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model

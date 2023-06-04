import torch
import re
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import LlamaTokenizer

class LlamaEngine(object):
    def __init__(self, llama_model_path, lora_model_path, use_lora=True):
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
        if use_lora:
            self.model = PeftModel.from_pretrained(model, lora_model_path)
        else:
            self.model = model
        self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)

    def generate(self, messages,
                 **generate_dict):
        """
        Args:
            messages: List of Dict
                [
                    {"role": "user", "text": "xxx"},
                    {"role": "", "text": "xxx"},
                ]
        Return:
            output_text: str
        """
        generation_config = dict(
            temperature=0.1,
            top_k=20,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.3,
            max_new_tokens=300)
        generation_config.update(generate_dict)
        pattern = "<|im_start|>{role}\n{text}\n<|im_end|>"
        items = []
        for message in messages:
            role, text = message['role'], message['text']
            item = pattern.format(role=role,
                                  text=text)
            items.append(item)
        input_text = "\n".join(items)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        output_ids = outputs[0][inputs['input_ids'].shape[1]:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        split_pattern = "<\|im_start\|>|<\|im_end\|>"
        for item in re.split(split_pattern, output_text):
            if item.strip():
                break
        extract_text = item.strip()
        if extract_text.startswith('assistant'):
            extract_text = extract_text[len('assistent'):].strip()
        return extract_text

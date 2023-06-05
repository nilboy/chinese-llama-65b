import fire
import glob
import gc
import os
import torch
from peft import LoraConfig
from peft.utils.other import transpose
from tqdm.auto import tqdm
gc.enable()

def merge_lora_model(lora_model_path,
                     base_model_path,
                     output_model_path):
    """
        Args:
            lora_model_path: str. lora模型
            base_model_path: str. 转换为hf的llama模型路径
            output_model_path: str. 合并后的llama模型路径
    """
    lora_config = LoraConfig.from_pretrained(lora_model_path)
    lora_dict = {}
    for pt_model in glob.glob(lora_model_path + '/*.bin'):
        model = torch.load(pt_model, map_location='cuda:0')
        for k, v in model.items():
            lora_dict['.'.join(k.split('.')[2:])] = v
        del model
        torch.cuda.empty_cache()
        gc.collect()
    os.system(f'cp -r {base_model_path} {output_model_path}')
    scaling = lora_config.lora_alpha / lora_config.r
    for base_model_part in tqdm(glob.glob(output_model_path + '/*.bin')):
        model = torch.load(base_model_part, map_location='cuda:0')
        for k, v in tqdm(model.items()):
            lora_k = '.'.join(k.split('.')[0:-1])
            if "embed_tokens" in lora_k:
                lora_a_name = ".lora_embedding_A"
                lora_b_name = ".lora_embedding_B"
                fan_in_fan_out = True
            else:
                lora_a_name = '.lora_A.weight'
                lora_b_name = '.lora_B.weight'
                fan_in_fan_out = lora_config.fan_in_fan_out
            if lora_k + lora_a_name not in lora_dict:
                continue
            lora_a = lora_dict[lora_k + lora_a_name]
            lora_b = lora_dict[lora_k + lora_b_name]
            v.data += transpose(lora_b @ lora_a, fan_in_fan_out) * scaling
        torch.save(model, base_model_part)


if __name__ == '__main__':
    fire.Fire(merge_lora_model)

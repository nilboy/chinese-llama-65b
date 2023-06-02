# chinese-llama-65b

基于[QLora](https://arxiv.org/abs/2305.14314), 在中文数据集上，继续预训练和指令微调LLaMA-65B模型。

# 转换模型, 扩充中文词表
将原始llama模型转化为hf格式，并且扩充词表，以及重新初始化新词embedding.
```
python tools/convert_llama_to_hf.py \
    --input_dir=/root/autodl-tmp/llama \
    --model_size=65B \
    --output_dir=/root/autodl-tmp/llama-hf/65B
```

## 训练数据格式

```
data/
├── test.jsonl
├── train.jsonl
└── val.jsonl
```
```
[
  {"role": "user", "text": "xxx"},
  {"role": "assistant", "text": "xxx"}
]
```

# 训练
```
accelerate launch tools/train.py conf/finetune_llama65b.yaml
```

# 合并lora和llama-65b模型
```
python tools/merge_lora_model.py \
    --lora_model_path /root/autodl-tmp/output-models-65B/checkpoint-550 \
    --base_model_path
    --output_model_path
```


# 推理
```
from chinese_qlora.engine import LlamaEngine

```

# 模型下载
Facebook官方发布的LLaMA模型禁止商用，为了遵循相应的许可，目前暂时无法发布完整的模型权重。
这里发布的是LoRA权重，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。
合并代码参考[merge_lora_model](tools/merge_lora_model.py)。

## 基于中英文维基百科继续预训练的LLaMA-65B模型
chinese-llama-65b-base

## 基于chinese-llama-65b-base进行指令微调的模型
chinese-llama-65b-instruct

## 系统评测

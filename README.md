# chinese-llama-65b

基于[QLora](https://arxiv.org/abs/2305.14314), 在中文数据集上，继续预训练和指令微调LLaMA-65B模型。

# 转换模型, 扩充中文词表
将原始llama模型转化为hf格式，并且扩充词表，以及重新初始化新词embedding.
```
python tools/convert_llama_to_hf.py \
    --input_dir==llama \
    --model_size=65B \
    --output_dir=llama-hf/65B
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
# 单GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py conf/finetune_llama65b.yaml
# 多GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/train.py conf/finetune_llama65b.yaml
```

# 合并lora和llama-65b模型
```
python tools/merge_lora_model.py \
    --lora_model_path output-models-65B/checkpoint-1000 \
    --base_model_path xxx
    --output_model_path xxx
```


# 推理
```
from chinese_qlora.engine import LlamaEngine

```

# 模型下载
Facebook官方发布的LLaMA模型禁止商用，为了遵循相应的许可，目前暂时无法发布完整的模型权重。
这里发布的是LoRA权重，可以理解为原LLaMA模型上的一个“补丁”，两者进行合并即可获得完整版权重。
合并代码参考[merge_lora_model](tools/merge_lora_model.py)。

## 基于llama-65b在中文数据继续预训练
| 模型名称   | chinese-llama-65b-base |
| --------- | --------- |
| 训练数据   | 中文wiki + 英文wiki (采样30k, 0.1B tokens) |
| 训练epoch   | 1  |
| lora权重   |   |

## 基于chinese-llama-65b-base进行指令微调的模型

| 模型名称   | chinese-llama-65b-instruct |
| --------- | --------- |
| 训练数据   | oasst1 + sharegpt + belle10m (平均随机采样40k)  |
| 训练epoch   | 1  |
| lora权重   |   |


# ⚠️ 局限性
1. 本项目目前使用的训练数据和计算资源都比较有限，模型效果还有待提高；
2. chinese-llama-65b-instruct模型在中文表现上远远弱于英文。 LLaMA-65B预训练过程中中文语料较少，虽然我们做了中文词表扩充，并在中英文wiki数据上继续做预训练，但是中文的效果仍然较差。
中文领域急需要一个好的LLM基座模型。
3. 我们后续会持续更新和发布模型；

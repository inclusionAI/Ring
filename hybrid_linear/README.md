# Ring-Linear

<p align="center">
    <img src="../figures/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>

## Introduction

Ring-lite-**linear**-preview is a hybrid-linear MoE LLM provided and open-sourced by InclusionAI, which has 17.1B parameters with 3.0B activated parameters. It is a long reasoning model based on hybrid-linear attention, achieving near-linear computational complexity and near-constant space complexity during inference. This model was converted from [Ling-lite-0220](https://huggingface.co/inclusionAI/Ling-lite/tree/Ling-lite-0220), which adopts the softmax attention-based architecture. It matches the performance of DeepSeek-R1-Distill-Qwen-7B on standardized reasoning benchmarks while substantially reducing computational overhead in both training and inference phases. In certain generation speed tests based on vLLM, we observed that the throughput was more than doubled compared to softmax attention models of the same scale (e.g., Ling-lite). To the best of our knowledge, it is the first open-source hybrid-linear reasoning language model.

## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<div align="center">

|     **Model**      | **#Total Params** | **#Activated Params** | **Context Length** | **Download** |
| :----------------: | :---------------: | :-------------------: | :----------------: | :----------: |
| Ring-lite-linear-preview |       17.1B       |         3.0B         |        64K         |      [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-lite-linear-preview)  <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ring-lite-linear-preview)  | 

</div>

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-lite-linear-preview"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ring, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### 🤖 ModelScope

If you're in mainland China, we strongly recommend you to use our model from 🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>.

## Deployment
### vLLM deployment

vLLM supports offline batched inference or launching an OpenAI-Compatible API Service for online inference.

#### Environment Preparation

Since the Pull Request (PR) has not been submitted to the vLLM community at this stage, please prepare the environment by following the steps below:

```bash
git clone -b  v0.7.3 https://github.com/vllm-project/vllm.git
cd vllm
git apply Ring/hybrid_linear/inference/vllm/bailing_moe_linear.patch
pip install -e .
```

#### Offline Inference:

```bash
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ring-lite-linear-preview")

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

llm = LLM(model="inclusionAI/Ring-lite-linear-preview", dtype='bfloat16')
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params)


```

We utilize YaRN in vLLM to handle long context by add a `rope_scaling` field to the `config.json` file of the model. For example,

```json
{
  ...,
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 16384,
    "type": "yarn"
  }
}
```

#### Online Inference:

```bash
vllm serve inclusionAI/Ring-lite-linear-preview \
              --tensor-parallel-size 2 \
              --pipeline-parallel-size 1 \
              --use-v2-block-manager \
              --gpu-memory-utilization 0.90
```

For detailed guidance, please refer to the vLLM [`instructions`](https://docs.vllm.ai/en/latest/).


## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring/blob/master/LICENSE).

## Citation

[TBD]

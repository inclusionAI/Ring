# Ring

<p align="center">
    <img src="./figures/ant-bailing.png" width="100"/>
<p>

<p align="center">
          🤗 <a href="https://huggingface.co/inclusionAI">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/inclusionAI">ModelScope</a>

## Introduction

Ring is a reasoning MoE LLM provided and open-sourced by InclusionAI, derived from [Ling](https://github.com/inclusionAI/Ling). We introduce two different sizes, which are Ring-Lite and Ring-Plus. Ring-Lite has 16.8 billion parameters with 2.75 billion activated parameters, while Ring-Plus has 290 billion parameters with 28.8 billion activated parameters. Both models demonstrate impressive reasoning performance compared to existing models in the industry.


## Model Downloads

You can download the following table to see the various parameters for your use case. If you are located in mainland China, we also provide the model on ModelScope.cn to speed up the download process.

<div align="center">

|      **Model**       | **#Total Params** | **#Activated Params** | **Context Length** |                                                                        **Download**                                                                        |
| :------------------: | :---------------: | :-------------------: | :----------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    Ring-lite-distill    |       16.8B       |         2.75B         |        64K         |     [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ring-lite-distill) <br>[🤖 ModelScope](https://modelscope.cn/models/inclusionAI/Ring-lite-distill)     |

</div>

## Quickstart

### 🤗 Hugging Face Transformers

Here is a code snippet to show you how to use the chat model with `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/Ring-lite-distill"

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
Please refer to [Ling](https://github.com/inclusionAI/Ling)

## Finetuning
Please refer to [Ling](https://github.com/inclusionAI/Ling)


## License

This code repository is licensed under [the MIT License](https://github.com/inclusionAI/Ring/blob/master/LICENSE).

## Citation

[TBD]

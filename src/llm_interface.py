from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

MODEL_NAME = "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"

print("START: Unsloth model load", flush=True)

model, processor = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
    text_only=True,
)

print("START: corrected tokenizer load", flush=True)

corrected_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    fix_mistral_regex=True,
    padding_side="left",
)

processor.tokenizer = corrected_tokenizer

print("Corrected tokenizer installed", flush=True)


def get_model():
    return model, processor.tokenizer

def query_llm(prompt: str):
    tokenizer = processor.tokenizer

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.get_input_embeddings().weight.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            use_cache=True,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
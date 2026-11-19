from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 1024, 
    load_in_4bit = True,
    text_only = True,
)

FastLanguageModel.for_inference(model)

def query_llm(prompt):
    messages = [
        {
            "role" : "user",
            "content" : prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        inputs, 
        max_new_tokens = 256,
        do_sample = False, 
        use_cache = True,
    )

    generated_tokens = outputs[0][inputs.shape[-1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens = True
    )

    return response
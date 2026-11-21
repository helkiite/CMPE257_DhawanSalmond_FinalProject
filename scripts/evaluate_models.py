import argparse
import torch
from datasets import load_dataset
from math_verify import parse, verify
from unsloth import FastLanguageModel

BASE_MODEL = "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit"

def get_answer(model, tokenizer, problem):
    prompt = f"""Solve this math problem. Return only: FINAL_ANSWER: <answer>   Problem: {problem}"""

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    )
    device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens = 64,
            do_sample = False,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens = True,
    )

    if "FINAL_ANSWER: " in response:
        return response.split("FINAL_ANSWER: ")[-1].strip()

    return response.strip()

def is_correct(gold, prediction):
    try:
        return bool(verify(parse(str(gold)), parse(str(prediction)),))
    except:
        return str(gold).strip() == str(prediction).strip()


def evaluate(model, tokenizer, dataset, problem_col, answer_col):
    correct = 0
    for i, example in enumerate(dataset):
        prediction = get_answer(model, tokenizer, example[problem_col],)
        gold = example[answer_col]

        if is_correct(gold, prediction):
            correct += 1
        print(f"{i + 1}/{len(dataset)} " f"Accuracy: {correct / (i + 1): .2%}")
        
    return correct / len(dataset)

def load_model(path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = path,
        max_seq_length = 1024,
        load_in_4bit = True,
    )

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def main(checkpoint):
    amc = load_dataset("math-ai/amc23", split = "test")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split = "test",)
    amc = amc.select(range(40))
    math500 = math500.select(range(100))

    print("\nBase Model")
    model, tokenizer = load_model(BASE_MODEL)

    base_amc = evaluate(model, tokenizer, amc, "question", "answer",)
    base_math = evaluate(model, tokenizer, math500, "problem", "answer",)

    del model
    del tokenizer
    torch.cuda.empty_cache()

    print("Checkpoint")
    model, tokenizer = load_model(checkpoint)

    checkpoint_amc = evaluate(model, tokenizer, amc, "question", "answer")
    checkpoint_math = evaluate(model, tokenizer, math500, "problem", "answer")

    print("\nResults")
    print(f"AMC: Base was {base_amc:.2%} and Checkpoint is {checkpoint_amc:.2%}")
    print(f"MATH-500: Base was {base_math:.2%} and Checkpoint is {checkpoint_math:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required = True,)

    args = parser.parse_args()
    main(args.checkpoint)


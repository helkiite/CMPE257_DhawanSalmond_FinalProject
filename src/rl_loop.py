from unsloth import FastLanguageModel
import torch

# immediate settings modifications to torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import llm_interface
import rl_functions
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint

# Training run size config
DATASET_SIZE: int = 10       # Number of problems to generate / for players to attempt.
NUM_GENERATIONS: int = 4    # Minimum 4. Number of player attempts at any given problem.

# Checkpoint config. Set True to continue from the highest checkpoint step.
RESUME_FROM_LATEST_CHECKPOINT: bool = True
OUTPUT_DIR: str = "outputs/player-grpo"

# Checks for valid checkpoint in OUTPUT_DIR
resume_checkpoint = None
if RESUME_FROM_LATEST_CHECKPOINT:
    if Path(OUTPUT_DIR).is_dir():
        resume_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        print(f"Resuming training from {resume_checkpoint}")
    if resume_checkpoint is None:
        raise FileNotFoundError(
            f"No training checkpoint found in {OUTPUT_DIR!r}. "
        )

# initialize model
model, tokenizer = llm_interface.get_model()

# generate dataset
print("Generating Coach Questions")
player_dataset = rl_functions.generate_player_dataset(DATASET_SIZE)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

FastLanguageModel.for_training(model)
model.print_trainable_parameters()

print("Device map:", getattr(model, "hf_device_map", None))
print("Parameter devices:", {str(p.device) for p in model.parameters()})

# initialize player rl config
training_args = GRPOConfig(
    output_dir="outputs/player-grpo",

    # Small batch
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=NUM_GENERATIONS,

    max_completion_length=512,
    temperature=0.7,
    top_p=0.9,

    learning_rate=2e-6,
    max_steps=DATASET_SIZE,

    # Keep task_id for custom reward function.
    remove_unused_columns=False,

    logging_steps=1,
    save_steps=20,
    report_to="none",

    use_vllm=False,
    beta=0.001,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rl_functions.majority_vote_reward,
    args=training_args,
    train_dataset=player_dataset,
)

print("Starting Training")
trainer.train(resume_from_checkpoint=resume_checkpoint)

# save model
model.save_pretrained("outputs/player-adapter")
tokenizer.save_pretrained("outputs/player-adapter")
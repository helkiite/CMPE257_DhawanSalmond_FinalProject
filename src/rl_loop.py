import unsloth
import torch
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import llm_interface
import rl_functions
from unsloth import FastLanguageModel

# Training run size config
DATASET_SIZE: int = 6       # Number of problems to generate / for players to attempt.
NUM_GENERATIONS: int = 4    # Minimum 4. Number of player attempts at any given problem.

# initialize model
model, tokenizer = llm_interface.get_model()

# generate dataset
print("Generating Coach Questions")
player_dataset = rl_functions.generate_player_dataset(DATASET_SIZE)

print("PLAYER DATASET")
print(player_dataset[:])

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

# initialize player rl config
training_args = GRPOConfig(
    output_dir="outputs/player-grpo",

    # Small batch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=NUM_GENERATIONS,

    max_completion_length=1024,
    temperature=0.7,
    top_p=0.9,

    learning_rate=1e-6,
    max_steps=DATASET_SIZE,

    # Keep task_id for custom reward function.
    remove_unused_columns=False,

    logging_steps=1,
    save_steps=int(DATASET_SIZE/2),
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
trainer.train()

# save model
model.save_pretrained("outputs/player-adapter")
tokenizer.save_pretrained("outputs/player-adapter")
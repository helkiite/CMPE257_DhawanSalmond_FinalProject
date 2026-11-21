from datasets import load_dataset


def load_amc():
    amc = load_dataset("math-ai/amc23", split = "test") #validation dataset from CP Mobius

    #printing AMC data details
    print("AMC rows: ", len(amc))
    print("AMC columns: ", amc.column_names)
    print("AMC first example: ")
    print(amc[0])
    return amc


def load_math500():
    math500 = load_dataset("HuggingFaceH4/MATH-500", split = "test") #MATH 500 dataset
    #printing MATH500 data details
    print("\nMATH500 rows: ", len(math500))
    print("MATH500 columns: ", math500.column_names)
    print("MATH500 first example: ")
    print(math500[0])
    return math500

def load_prime():
    prime = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train") # Dataset used for coach warmup
    return prime

# Prepare and format dataset for coach warmup.
def prepare_prime(number_of_examples: int = 4000, seed_value: int = 67):
    prime = load_prime()

    # setup filter for nonvalid examples
    def has_math_problem(example):
        if example["ability"] != "math":
            return False

        return any(
            message["role"] == "user"
            and isinstance(message["content"], str)
            and message["content"].strip()
            for message in example["prompt"]
        )

    # apply filter
    filtered_prime = prime.filter(has_math_problem)

    # check we have enough examples
    if len(filtered_prime) < number_of_examples:
        raise ValueError(
            f"Only {len(filtered_prime)} usable math examples were found; "
            f"{number_of_examples} are required."
        )

    # set seed for reproducibility
    prime_subset = filtered_prime.shuffle(seed=seed_value).select(
        range(number_of_examples)
    )

    def format_for_coach(example):
        """Turn a PRIME math question into a Coach training example."""

        # PRIME stores the question as the user message in its prompt.
        problem = next(
            message["content"].strip()
            for message in example["prompt"]
            if message["role"] == "user"
            and isinstance(message["content"], str)
            and message["content"].strip()
        )

        # TODO: PRIME examples may end with an instruction like 'answer in __ format'. Ideally I would remove this.

        # return formatted outputs
        return {
            "messages": [
                {
                "role": "user",
                "content": ("You are the Coach in a reinforcement learning system. Generate one mathematical reasoning problem for the Player to solve. The problem should have a clear and verifiable final answer. Return only the problem."),
                },
                {
                    "role": "assistant",
                    "content": problem,
                },
            ]
        }


    # Keep only the conversational training column.
    coach_dataset = prime_subset.map(
        format_for_coach,
        remove_columns=prime_subset.column_names,
    )


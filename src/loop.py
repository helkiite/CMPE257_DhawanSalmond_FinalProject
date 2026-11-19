from collections import Counter
from llm import query_llm

def majority_vote(answers): #return most popular answer
    counts = Counter(answers)
    majority_answers = counts.most_common(1)[0][0]
    return majority_answers

def consistency_score(answers, majority_answer): #how many answers are equal to the majority
    matching_answers = sum(
        1 for answer in answers
        if answer == majority_answer
    )

    return matching_answers / len(answers)

def player_reward(answers, majority_answer): #1 if matches majority 0 elsewise
    rewards = [
        1 if answer == majority_answer 
        else 0
        for answer in answers
    ]

    return sum(rewards) / len(rewards)

def coach_reward(player_reward_value, delta_accuracy): #reward = player reward * validation improvement
    return player_reward_value * delta_accuracy

def generate_coach_problem(): #return one math problem
    prompt = """ You are the Coach in a reinforcement learning system. Generate one mathematical reasoning problem for the Player to solve. The problem should have a clear and verifiable final answer. Return only the problem. """
    problem = query_llm(prompt)
    return problem

#m = num of accepted coach problems in a batch
#n = number of player attempts per problem
#T = number of coach-player interactions
#D_val = validation dataset, like AMC

def generate_player_responses(problem, n): #return list of responses to the prompt
    responses = []
    for i in range(n):
        prompt = f""" Solve the following math problem. Show your reasoning and state your final answer clearly. End your responses with exactly: FINAL_ANSWER: <answer>. Problem: {problem} """
        response = query_llm(prompt)
        responses.append(response)

    return responses
    
def extract_final_answer(response):
    marker = "FINAL_ANSWER: "
    if marker in response:
        return response.split(marker)[-1].strip()

    return response.strip()

def build_batch(m, n): #the accepted batch
    accepted_batch = []
    attempts = 0
    max_attempts = 100

    while len(accepted_batch) < m and attempts < max_attempts:
        attempts += 1
        problem = generate_coach_problem()
        responses = generate_player_responses(problem, n)

        answers = [
            extract_final_answer(response)
            for response in responses
        ]

        majority_answer = majority_vote(answers)
        consistency =  consistency_score(answers, majority_answer)

        if 0.2 <= consistency <= 0.8:
            reward = player_reward(answers, majority_answer)
            accepted_batch.append({
                "problem": problem, 
                "responses": responses,
                "answers": answers, 
                "majority_answer": majority_answer,
                "consistency": consistency,
                "player_reward": reward
            })
    return accepted_batch
    

def run_loop(T, m, n):
    pass

#test_answers = ["4", "5", "7", "7", "7", "4"]
#majority = majority_vote(test_answers)
#consistency = consistency_score(test_answers, majority)
#p_reward = player_reward(test_answers, majority)
#c_reward = coach_reward(p_reward, 0.08)

#print("m: ", majority)
#print("c: ", consistency)
#print("p rew: ", p_reward)
#print("c rew: ", c_reward)

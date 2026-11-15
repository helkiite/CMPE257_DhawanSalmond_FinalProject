# CMPE257_DhawanSalmond_FinalProject
This is our implementation of the CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning.

Link to paper: https://openreview.net/forum?id=JlX4RDitRr

Initial Approach
We will implement the paper's multi-agent reinforcement learning loop. It works by instantiating a Coach and a Player model. The Coach proposes problems of which the Player attempts several times. Agreement amongst answers is used as a proxy for difficulty and correctness - no agreement amongst answers is seen as too difficult, perfect agreement means the problem is too easy. The majority consensus amongst medium-difficulty questions are then used in a fine-tuning regime to improve the player model. The Player model is then evaluated against a withheld validation set. If the Player model improves its scores, then the Coach model is fine-tuned on the questions it provided - making it more likely to repeat the teaching techniques that found success. If the Player model finds lower success in its scores - then the Coach model is penalized in order to avoid that instructive methodology in future rounds.

With this recursive feedback loop, both models slowly improve and maximize the Player’s score. For our project, we plan to broadly follow the author’s original methodology and use mathematics as the target domain. We will use the mathematical dataset described in the paper, performing any data cleaning the authors advised. We will fine tune models using QLoRA. If we find it necessary to shard the training into parallel streams, we will unify weights with a SLERP merge.

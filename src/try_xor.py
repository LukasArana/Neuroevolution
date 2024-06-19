from cma_strat import cma_strat
from neat_strat import neat_strat
from random_strat import random_strat
import numpy as np
import time
import sys
import neat
import gymnasium as gym


def get_config():
    path = "configs/config"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, #nn information from Neat configuration file
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            path)

    #Define input, ouput and hidden values
    config.genome_config.num_inputs = 2
    config.genome_config.num_hidden = 0
    config.genome_config.num_outputs = 1
    config.genome_config.input_keys = list(reversed(range((config.genome_config.num_inputs * -1), 0))) #[-1, ..., -num_inputs]
    config.genome_config.output_keys = list(range(0, config.genome_config.num_outputs)) #[1, ..., num_outputs]
    return config

def evaluate_xor( policy_nn, nreps, seed, record = False, path = None, var = False):
    xor_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    xor_outputs = np.array([0.0, 1.0, 1.0, 0.0])

    rewards_reps = np.zeros(nreps)
    for rep_idx in range(nreps):
        total_reward = 0

        for i in range(4):  # for each XOR input
            observation = xor_inputs[i]
            output = policy_nn.get_output(observation)
            #action = np.round(output)  # round to get binary output
            reward = -abs(xor_outputs[i] - output)  # reward is 1 for correct action, -1 for incorrect
            total_reward += reward
        rewards_reps[rep_idx] = total_reward

    if var:
        return np.mean(rewards_reps), np.var(rewards_reps)
    return np.mean(rewards_reps)

env =  gym.make('InvertedDoublePendulum-v4', render_mode='rgb_array')
SEED=3
rs = np.random.RandomState(seed=SEED)
if __name__ == "__main__":
    max_evals = 50000
    print(f"ENV = XOR")
    nreps = 20 # Number of repetitions for each algorithm in each env
    for rep_idx in range(nreps):
        config = get_config()

        arch_seed = rs.randint(int(1e8))
        strats = [cma_strat(arch_seed, config, "cma")] 

        for strat in strats:
            best = (-sys.maxsize - 1, None)
            start_time = time.time()
            for evaluation_idx in range(max_evals):
                nn = strat.show()
                f = evaluate_xor(nn, 1, rs.randint(int(1e8))) # Posible da 1 ordez 20 jartzea
                strat.tell(f)
                if f > best[0]:
                    best = (f, nn)
                if (evaluation_idx + 1) % 1000 == 0 or evaluation_idx == 0:
                    nn = best[1]
                    f = evaluate_xor(nn, 100, 2) # Test seed always the same
                    best = (f, nn)
                    print(evaluation_idx / (max_evals+1), f)
                    strat.log(f"results/data/pruebaRandom/HiddenXOR/{strat.name}_XOR_{SEED}_{rep_idx}.txt", f, nn, evaluation_idx+1, 1, time.time() - start_time)
                    fs = []
            print(f"time for all evaluations: {time.time() - start_time}")

import gymnasium as gym
from cma_strat import cma_nn, cma_strat
from neat_strat import neat_nn, neat_strat
from interfaces import policy_nn, optimization_strat
import numpy as np
import time
import cv2
import neat
import argparse
import numbers
import pickle
import sys

def get_config(path = "/home/walle/Desktop/TFG/nofn/configs/config"):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, #nn information from Neat configuration file
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            path)

    if np.issubdtype(env.action_space.dtype, np.integer): #Check output if is discrete
        action_space_dim = int(env.action_space.n)
    else:
        action_space_dim = 1 # Action space = 1 is assummed in continious

    #Define input, ouput and hidden values
    config.genome_config.num_inputs = int(env.observation_space.shape[0])
    config.genome_config.num_hidden = 0
    config.genome_config.num_outputs = action_space_dim
    config.genome_config.input_keys = list(reversed(range((config.genome_config.num_inputs * -1), 0))) #[-1, ..., -num_inputs]
    config.genome_config.output_keys = list(range(0, config.genome_config.num_outputs)) #[1, ..., num_outputs]
    return config

def evaluate_policy(policy_nn: policy_nn, nreps, seed):
    rewards_reps = np.zeros(nreps)
    for rep_idx in range(nreps):
        observation, info = env.reset(seed=seed)
        terminated, truncated = False, False
        total_reward = 0
        while not (terminated or truncated): 
            output = policy_nn.get_output(observation)
            if np.issubdtype(env.action_space.dtype, np.integer): # check if is discrete
                action = np.argmax(output)
            else:
                action = output * env.action_space.high # Scale number from [-1, 1] to the action_space 
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward # Acumulated reward
        rewards_reps[rep_idx] = total_reward

    return np.mean(rewards_reps)



ENVS = {"pendulum": gym.make('Pendulum-v1'),
            "mountain_car_cont": gym.make('MountainCarContinuous-v0'),
            "mountain_car": gym.make("MountainCar-v0"),
            "lunar": gym.make( "LunarLander-v2"),
            "cart": gym.make("CartPole-v1"),
            "acrobot": gym.make("Acrobot-v1")}
ENVS = {"mountain_car_cont": gym.make('MountainCarContinuous-v0'),
            "mountain_car": gym.make("MountainCar-v0"),
            "lunar": gym.make( "LunarLander-v2"),
            "cart": gym.make("CartPole-v1"),
            "acrobot": gym.make("Acrobot-v1")}

SEED=2
rs = np.random.RandomState(seed=SEED)
if __name__ == "__main__":
    max_evals = 20000

    for env_name in ENVS:
        print(f"ENV = {env_name}")
        env = ENVS[env_name]
        config = get_config()
        nreps = 20 # Number of repetitions for each algorithm in each env
        for rep_idx in range(nreps):
            arch_seed = rs.randint(int(1e8))
            strats = [cma_strat(arch_seed, config, "cma"), neat_strat(arch_seed, config, "neat")] 
            for strat in strats:
                start_time = time.time()
                best = (-sys.maxsize - 1, None)
                for evaluation_idx in range(max_evals):
                    nn = strat.show()
                    f = evaluate_policy(nn, 1, rs.randint(int(1e8))) # Posible da 1 ordez 20 jartzea
                    strat.tell(f)

                    if f > best[0]:
                        best = (f, nn)

                    if evaluation_idx % 1000 == 0:
                        f = evaluate_policy(best[1], 100, rs.randint(int(1e8))) # 100 aldiz eta bere media
                        print(evaluation_idx / max_evals, f)
                        strat.log(f"../nofn/results/data/final/{env_name}/{strat.name}_{env_name}_{SEED}_{rep_idx}.txt", f, evaluation_idx+1, env._elapsed_steps, time.time() - start_time)
                        fs = []
                env.close()
import gymnasium as gym
from cma_strat import cma_nn, cma_strat
from neat_strat import neat_nn, neat_strat
from random_strat import random_strat
from interfaces import policy_nn, optimization_strat
import numpy as np
import time
import cv2
import neat
import argparse
import numbers
import pickle
import sys
def get_config(env, path = "/home/walle/Desktop/TFG/nofn/configs/config"):
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

def evaluate_policy(policy_nn: policy_nn, nreps, seed, env, record = False, path = None, var = False):
    video_name = path
    rewards_reps = np.zeros(nreps)
    rs = np.random.RandomState(seed = seed)
    for rep_idx in range(nreps):
        observation, info = env.reset(seed=rs.randint(int(1e8)))
        terminated, truncated = False, False
        total_reward = 0
        episode_frame = 0
        while not (terminated or truncated): 
            
            output = policy_nn.get_output(observation)
            if np.issubdtype(env.action_space.dtype, np.integer): # check if is discrete
                action = np.argmax(output)
            else:
                action = output * env.action_space.high # Scale number from [-1, 1] to the action_space
            observation, reward, terminated, truncated, info = env.step(action)
            if record:
                render = env.render()
                if episode_frame == 0:
                    height, width, layers = render.shape
                    video = cv2.VideoWriter(video_name,0, 30, (width, height))
                video.write(render)
                episode_frame += 1
            total_reward += reward # Acumulated reward
        env.close()

        rewards_reps[rep_idx] = total_reward

        if record:
            cv2.destroyAllWindows()
            video.release()
    if var:
        return np.mean(rewards_reps), np.var(rewards_reps)
    return np.mean(rewards_reps)


"""
ENVS = {"cart": gym.make("CartPole-v1", render_mode = "rgb_array"),
        "pendulum": gym.make('Pendulum-v1'),
        "mountain_car_cont": gym.make('MountainCarContinuous-v0'),
        "mountain_car": gym.make("MountainCar-v0"),
        "lunar": gym.make( "LunarLander-v2"),
        "acrobot": gym.make("Acrobot-v1"),
        "DoubleInvertedPendulum": gym.make('InvertedDoublePendulum-v4'),
        "InvertedPendulum": gym.make("InvertedPendulum-v4")}
"""
ENVS = {"pendulum": gym.make('Pendulum-v1')}
#ENVS = {"mountain_car": gym.make("MountainCar-v0")}
#ENVS = {"DoubleInvertedPendulum": gym.make('InvertedDoublePendulum-v4')}
SEED=3
rs = np.random.RandomState(seed=SEED)
if __name__ == "__main__":
    max_evals = 20000
    for env_name in ENVS:
        print(f"ENV = {env_name}")
        env = ENVS[env_name]
        config = get_config(env)
        nreps = 20 # Number of repetitions for each algorithm in each env
        for rep_idx in range(nreps):
            arch_seed = rs.randint(int(1e8))
            strats = [neat_strat(arch_seed, config, "neat"), cma_strat(arch_seed, config, "cma"), random_strat(arch_seed, config, "random")] 
            for strat in strats:
                best = (-sys.maxsize - 1, None)
                start_time = time.time()
                for evaluation_idx in range(max_evals):
                    nn = strat.show()
                    f = evaluate_policy(nn, 1, rs.randint(int(1e8)), env) # Posible da 1 ordez 20 jartzea
                    strat.tell(f)
                    if f > best[0]:
                        #print(f, evaluate_policy(nn, 100, SEED, env) )
                        best = (f, nn)
                    if evaluation_idx % 1000 == 0:
                        nn = best[1]
                        f = evaluate_policy(nn, 100, 2, env) # Test seed always the same
                        print(evaluation_idx / max_evals, f)
                        strat.log(f"results/data/pruebaRandom/{env_name}/{strat.name}_{env_name}_{SEED}_{rep_idx}.txt", f, nn, evaluation_idx+1, env._elapsed_steps, time.time() - start_time)
                        fs = []
                print(f"time for all evaluations: {time.time() - start_time}")

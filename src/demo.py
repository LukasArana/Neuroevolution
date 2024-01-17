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

def _parse_args():
    CONFIG_FILENAME = ""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', nargs='?', default=CONFIG_FILENAME, help='Configuration filename') #Specify NEAT config file

    command_line_args = parser.parse_args()

    CONFIG_FILENAME = command_line_args.config
    return command_line_args

command_line_args = _parse_args()
conf_name = command_line_args.config


envs = {"pendulum": gym.make('Pendulum-v1', g=9.81),
        "mountain_car": gym.make('MountainCarContinuous-v0', render_mode="rgb_array"),
        "lunar": gym.make( "LunarLander-v2",
                                continuous = False,
                                gravity = -10.0,
                                enable_wind = False,
                                wind_power = 15.0,
                                turbulence_power = 1.5,
                                render_mode="rgb_array"),
        "cart": gym.make("CartPole-v1")
}

env_names = list(envs.keys())

env_name = env_names[1] #Choose number
env = envs[env_name]

seed=2

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, #nn information from Neat configuration file
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        conf_name)


strat = neat_strat(seed, config, "neat")
#strat = cma_strat(seed, config, "cma")



def evaluate_policy(policy_nn: policy_nn, nreps, seed, show = False):
    rewards_reps = np.zeros(nreps)
    rs = np.random.RandomState(seed = seed)
    if show: 
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter("video.avi", fourcc, 20.0, (400,600))
    for rep_idx in range(nreps):
        observation, info = env.reset(seed=rs.randint(int(1e8)))
        terminated, truncated = False, False
        total_reward = 0

        while not (terminated or truncated): 
            observation = (observation - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) #MinMax to 0-1 range
            output = policy_nn.get_output(observation)
            if is_discrete:
                action = np.argmax(output)
            else:
                action = output * env.action_space.high # Scale number from [-1, 1] to the action_space 
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if show:
                video.write(env.render())
        rewards_reps[rep_idx] = total_reward

    if show:
        cv2.destroyAllWindows()
        video.release()
    return np.mean(rewards_reps)


assert len(env.observation_space.shape) == 1, "observation space needs to be flat" 



is_discrete = np.issubdtype(env.action_space.dtype, np.integer)

observation_space_dim = int(env.observation_space.shape[0])
assert observation_space_dim == config.genome_config.num_inputs , "Config file input number and observatioin space len must be the same" 

#print(bool(gym.spaces.discrete.Discrete))
if is_discrete: #Check output if is discrete
    action_space_dim = int(env.action_space.n)
else:
    action_space_dim = 1 # Action space = 1 is assummed in continious
assert action_space_dim == config.genome_config.num_outputs, "Config file output number and action space len must be the same" 



max_evals = 20000
rs = np.random.RandomState(seed=seed)
start_time = time.time()
fs = []
for evaluation_idx in range(max_evals):
    nn = strat.show()
    f = evaluate_policy(nn, 1, rs.randint(int(1e8)))
    strat.tell(f)
    fs.append(f)
    if evaluation_idx % 1000 == 0:
        f = evaluate_policy(nn, 1000, seed) # Test seed always the same
        #evaluate_policy(nn, 1, seed, show=True) # Saving video
        print(f"Avg : {np.mean(fs)}, VAR: {np.var(fs)}")
        print(evaluation_idx / max_evals, f)
        strat.log(f"../nofn/results/data/{strat.name}_{env_name}_{seed}.txt", f, evaluation_idx+1, env._elapsed_steps, fs, time.time() - start_time)
        fs = []
        
env.close()


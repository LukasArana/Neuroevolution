import numpy as np
from cmaes import CMA
import cma
from cma_strat import cma_nn
from main import evaluate_policy
import gymnasium as gym
import neat
import random
import sys
import time
from random_strat import random_strat
from neat_strat import neat_nn, neat_strat


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



ENVS = {"cart": gym.make("CartPole-v1", render_mode = "rgb_array"),
        "pendulum": gym.make('Pendulum-v1'),
        "mountain_car_cont": gym.make('MountainCarContinuous-v0'),
        "mountain_car": gym.make("MountainCar-v0"),
        "lunar": gym.make( "LunarLander-v2"),
        "acrobot": gym.make("Acrobot-v1"),
        "DoubleInvertedPendulum": gym.make('InvertedDoublePendulum-v4'),
        "InvertedPendulum": gym.make("InvertedPendulum-v4")}

SEED=3
rs = np.random.RandomState(seed=SEED)
n_reps = 20
max_evals = 15000
if __name__ == "__main__":
    for env_name in ENVS:
        print(f"ENV = {env_name}")

        env = ENVS[env_name]
        config = get_config(env)
        nn = cma_nn(config.genome_config.num_inputs, config.genome_config.num_outputs)
        strat = neat_strat(2, config, "neat")
        for rep_idx in range(n_reps):
            params =  np.random.normal(0, 1, cma_nn(config.genome_config.num_inputs, config.genome_config.num_outputs)._get_total_dim())
            arch_seed = rs.randint(int(1e8))
            optimizer = CMA(mean=params, sigma=1.3, seed = arch_seed)
            best = (-sys.maxsize - 1, None)
            start_time = time.time()

            for generation in range(max_evals // optimizer.population_size):
                solutions = []
                for i in range(optimizer.population_size):

                    evaluation_idx = (generation * optimizer.population_size + i)
                    x = optimizer.ask()
                    nn.set_parameters(x)
                    f = evaluate_policy(nn, 1, rs.randint( int(1e8)), env) # Posible da 1 ordez 20 jartzea
                    solutions.append((x, -f))
                    if f > best[0]:
                        best = (f, nn)
                    if evaluation_idx % 1000 == 0:
                        nn = best[1]
                        f = evaluate_policy(nn, 100, 100, env) # Test seed always the same
                        print(evaluation_idx / max_evals, f)
                        print(best[0])
                        strat.log(f"results/data/pruebaRandom/{env_name}/newCMA_{env_name}_{SEED}_{rep_idx}.txt", f, nn, evaluation_idx+1, env._elapsed_steps, time.time() - start_time)
                optimizer.tell(solutions)

"""
if __name__ == "__main__":
    es = cma.CMAEvolutionStrategy(params, 1.3, {"popsize": 10})

    for generation in range(2000):
        sol = []
        solutions = es.ask()
        for i in solutions:
            nn.set_parameters(i)
            sol.append(nn.get_output(noise))
        value = [quadratic(*i) for i in sol]
        es.tell(solutions, value)
        print(f"#{generation} {np.mean(value)}")
"""
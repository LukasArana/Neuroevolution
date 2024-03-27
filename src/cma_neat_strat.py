import glob
import pickle
import os
import neat
import numpy as np
import gymnasium as gym
from collections import defaultdict, deque
from main import evaluate_policy, get_config
from cma_strat import cma_nn, cma_strat, cma_hyperparams
import cma
import sys
import time
from cma_strat import cma_nn, cma_strat
from random_strat import random_strat
import copy
def get_nns(path):
    nns_neat = {i :[] for i in range(20)}
    idx_cma = 0
    idx_neat = 0
    for idx, i in enumerate(glob.glob(f"{path}/neat*.pkl")):
        objects = []
        with open(i, "rb") as f:
            while True: #w - 376, h - 400
                try:
                    obj = pickle.load(f)
                    objects.append(obj)
                except EOFError:
                    break
        if "neat" in i:
            nns_neat[idx_neat] = objects
            idx_neat += 1
    for i in range(3):
        path, _ = os.path.split(path)

    return nns_neat

class neat_cma(cma_nn):
    def __init__(self,neat_nn):
        self.n_in = len(neat_nn.input_nodes)
        self.n_out = len(neat_nn.output_nodes)
        self.n_middle = len([i[0] for i in neat_nn.node_evals if (not (i[0] < 0 or i[0] in (np.arange(self.n_out) )))])
        self.neat_nn = neat_nn
        
        self.n_neurons = self.n_in + self.n_out + self.n_middle
        self.n_weights = sum([len(i[5]) for i in neat_nn.node_evals]) + self.n_neurons # weight + biases
        self.weights = np.zeros(self.n_weights) #Initialize weights to 0
        self.biases = np.zeros(self.n_neurons) #Initialize bias to 0
        self.conns = self.get_connections()

        self.values = {}
        
        self.neuron_idx = neat_nn.input_nodes + neat_nn.output_nodes
        self.neuron_idx += [i for i in self.conns if i not in self.neuron_idx]
        self.neuron_idx = {i:idx for idx, i in enumerate(self.neuron_idx)}     #Translate neat idx to [0,1, 2, 3...] 

    #Creates the connections array for the feedForward
    def get_connections(self):
        idx_w = 0
        conns = {}    
        for i in self.neat_nn.node_evals:  # Iterate between hidden and output neurons
            idx = i[0]
            act_fun = i[1]
            agg_fun = i[2]
            bias = i[3]
            response = i[4]
            assert 0 <= response <= 1
            connection = [(j[0], self.weights[idx_w + idx_w2]) for idx_w2, j in enumerate(i[5])]
            conns[idx] = connection
            idx_w += len(i[5])
        return conns
        
    def get_output(self, input_vals):
        self.conns = self.get_connections()
        out_neurons = np.arange(self.n_out)
        input_neurons = (np.arange(self.n_in) + 1) * -1

        assert len(input_neurons) == len(input_vals)
       
        self.values = {j: i for i, j in zip(input_vals, input_neurons)} # Restart values

        for i in out_neurons: # Feed Forward
            if not i in self.conns: # Output neuron without conenctions
                self.values[i] = 0.0
                continue
            self.values[i] = self.forward(i, self.conns[i])

        return [self.values[i] for i in range(self.n_out)]

    def forward(self, idx, connection_nueron):
        val = 0
        for i in connection_nueron: # connections to the actual neuron
            if i[0] in self.values: # If the value has already been calculated
                val += self.values[i[0]] * i[1]
            else: # If the value has to be calcualted
                val +=  self.forward(i[0], self.conns[i[0]]) *i[1]

        return np.tanh(val + self.biases[self.neuron_idx[idx]])
    

class cma_neat_strat(cma_strat):

    strat_name = "cma_neat"
    es = None
    _solution_idx = 0

    def __init__(self, seed, config, name, neat_nn):
        self._solutions = []
        self._f_values = []
        self.seed = seed
        self.name = name
        self.n_in = config.genome_config.num_inputs 
        self.n_out = config.genome_config.num_outputs
        self.random_state = np.random.RandomState(seed)
        self.cma_nn = neat_cma(neat_nn)

        x0 = self.random_state.normal(cma_hyperparams.mean_normal_initialization, cma_hyperparams.std_normal_initialization, self.cma_nn._get_total_dim())
        self.es = cma.CMAEvolutionStrategy(x0, cma_hyperparams.cma_sigma0, {"seed":seed, "popsize": cma_hyperparams.pop_size, 'bounds': [[-1], [1]]})
        self._solutions = self.es.ask()

    def show(self) : # Return a nn
        if self._solution_idx >= len(self._solutions):
            #print(self._f_values)
            self.es.tell(self._solutions, self._f_values)
            self._solutions = self.es.ask()
            self._solutions = self.random_state.normal(cma_hyperparams.mean_normal_initialization, cma_hyperparams.std_normal_initialization, (cma_hyperparams.pop_size, self.cma_nn._get_total_dim()))
            self._f_values = []
            self._solution_idx = 0
        self.cma_nn.set_parameters(self._solutions[self._solution_idx])
        
        return self.cma_nn


ENVS = {"cart": gym.make("CartPole-v1", render_mode = "rgb_array"),
        "pendulum": gym.make('Pendulum-v1'),
        "mountain_car_cont": gym.make('MountainCarContinuous-v0'),
        "mountain_car": gym.make("MountainCar-v0"),
        "lunar": gym.make( "LunarLander-v2"),
        "acrobot": gym.make("Acrobot-v1"),
        "DoubleInvertedPendulum": gym.make('InvertedDoublePendulum-v4'),
        "InvertedPendulum": gym.make("InvertedPendulum-v4")}
def main():
    SEED=3
    rs = np.random.RandomState(seed=SEED)

    if __name__ == "__main__":
        max_evals = 20000
        for env_name in ENVS:
            env = ENVS[env_name]
            print(env_name)
            config = get_config(env)
            neat_nns = get_nns(f"/home/walle/Desktop/TFG/nofn/results/data/pruebaRandom/{env_name}")
            neat_nns = [i[-1] for i in neat_nns.values()] # Take the fully trained ones
            for rep_idx, arch in enumerate(neat_nns):
                arch_seed = rs.randint(int(1e8))
                strats = [cma_neat_strat(arch_seed, config, "cma_neat", arch.nn)] 
                #strats = [random_strat(arch_seed, config, "FullyRandom", True)] 
                for strat in strats:
                    best = (-sys.maxsize - 1, None)
                    start_time = time.time()
                    for evaluation_idx in range(max_evals):
                        nn = strat.show()
                        f = evaluate_policy(nn, 1, rs.randint(int(1e8)), env) # Posible da 1 ordez 20 jartzea
                        strat.tell(f)
                        if f > best[0]:
                            best = (f, copy.deepcopy(nn))
                        if evaluation_idx % 1000 == 0:
                            nn = best[1]
                            f = evaluate_policy(nn, 100, 2, env) # Test seed always the same
                            best = (f, nn)

                            print(evaluation_idx / max_evals, f)
                            strat.log(f"results/data/pruebaRandom/{env_name}/{strat.name}_{env_name}_{SEED}_{rep_idx}.txt", f, nn, evaluation_idx+1, env._elapsed_steps, time.time() - start_time)
                            fs = []
                    print(f"time for all evaluations: {time.time() - start_time}")

main()

"""  
"PROBA"
if __name__ == "__main__": 
    
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


    nn = neat_cma(neat_nn[15][19].nn)
    print(nn.get_output([0, 0, 0, 0, 0, 0]))
    print(nn._get_total_dim())
"""
from interfaces import *
import cma
import neat
import argparse
from cma_strat import cma_nn, cma_hyperparams
import gymnasium as gym

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

class random_es:
    def __init__(self, rs, dim):
        self.random_state = rs
        self.dim = dim
    def ask(self):
        return self.random_state.normal(cma_hyperparams.mean_normal_initialization, cma_hyperparams.std_normal_initialization, (cma_hyperparams.pop_size, self.dim)).reshape((cma_hyperparams.pop_size, self.dim))
    def tell(self, a, b):
        pass

class random_strat(optimization_strat):

    strat_name = "random"
    es = None
    _solution_idx = 0

    def __init__(self, seed, config, name):
        self._solutions = []
        self._f_values = []
        self.seed = seed
        self.name = name
        self.n_in = config.genome_config.num_inputs 
        self.n_out = config.genome_config.num_outputs
        self.random_state = np.random.RandomState(seed)
        self.es = random_es(self.random_state, cma_nn(self.n_in, self.n_out)._get_total_dim())
        x0 = self.random_state.normal(cma_hyperparams.mean_normal_initialization, cma_hyperparams.std_normal_initialization, cma_nn(self.n_in, self.n_out)._get_total_dim())
        self._solutions = self.es.ask()
        self.cma_nn = cma_nn(self.n_in, self.n_out)
        self.cma_nn.set_parameters(x0)
        
    def show(self) -> policy_nn: # Return the best nn found
        if self._solution_idx >= len(self._solutions):

            self.es.tell(self._solutions, self._f_values)
            self._solutions = self.es.ask()
            self._f_values = []
            self._solution_idx = 0
            
        res = cma_nn(self.n_in, self.n_out)
        res.set_parameters(self._solutions[self._solution_idx])
        return res

    def tell(self, f: float) -> None: # Update ith the reward info
        self._f_values.append(-f) # we want to maximize the total reward, and cma-es is minimizing.
        self._solution_idx += 1
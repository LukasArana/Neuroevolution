from interfaces import *
import cma
import neat
import argparse

class cma_hyperparams:
    mean_normal_initialization=0
    std_normal_initialization=0.1
    cma_sigma0=0.2
    n_middle_layers=3# it is actually the number of mid-layer to mid-layer weight-sets
    n_params_middle_layers=7



class cma_nn:

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        self.weights.append(np.zeros((cma_hyperparams.n_params_middle_layers, n_in), dtype=np.float32))
        self.biases.append(np.zeros((cma_hyperparams.n_params_middle_layers,), dtype=np.float32))
        
        for _ in range(cma_hyperparams.n_middle_layers):
            self.weights.append(np.zeros((cma_hyperparams.n_params_middle_layers, cma_hyperparams.n_params_middle_layers), dtype=np.float32))
            self.biases.append(np.zeros((cma_hyperparams.n_params_middle_layers,), dtype=np.float32))
            
        self.weights.append(np.zeros((n_out, cma_hyperparams.n_params_middle_layers), dtype=np.float32))
        self.biases.append(np.zeros((n_out,), dtype=np.float32))


    def get_output(self, input): 
        assert len(input) == self.n_in
        assert input.dtype == np.dtype('float32'), f"instead, input.dtype = {input.dtype}"

        prev = input.copy()
        
        for i in range(len(self.weights)):
            prev = np.tanh(np.dot(prev, self.weights[i].T) + self.biases[i]) 
        return prev

    def set_parameters(self, flat_parameters):
        index = 0
        # Set weights from flat_parameters
        for i in range(len(self.weights)):
            shape = self.weights[i].shape
            size = self.weights[i].size
            self.weights[i] = flat_parameters[index:index + size].reshape(shape)
            index += size
        
        # Set biases from flat_parameters
        for i in range(len(self.biases)):
            shape = self.biases[i].shape
            size = self.biases[i].size
            self.biases[i] = flat_parameters[index:index + size].reshape(shape)
            index += size

    def _get_total_dim(self):
        res = 0
        for w in self.weights:
            res += w.size        
        for b in self.biases:
            res += b.size
        return res 




class cma_strat(optimization_strat):

    strat_name = "cma"
    es = None
    _solutions = []
    _f_values = []
    _solution_idx = 0

    def __init__(self, seed, config):
 
        self.n_in = config.genome_config.num_inputs 
        self.n_out = config.genome_config.num_outputs
        self.random_state = np.random.RandomState(seed)
        x0 = self.random_state.normal(cma_hyperparams.mean_normal_initialization, cma_hyperparams.std_normal_initialization, cma_nn(self.n_in, self.n_out)._get_total_dim())
        self.es = cma.CMAEvolutionStrategy(x0, cma_hyperparams.cma_sigma0, {"seed":seed})
        self._solutions = self.es.ask()


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

"""
def _parse_args():
    CONFIG_FILENAME = ""
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', nargs='?', default=CONFIG_FILENAME, help='Configuration filename') #Specify NEAT config file

    command_line_args = parser.parse_args()

    CONFIG_FILENAME = command_line_args.config
    return command_line_args

command_line_args = _parse_args()
conf_name = command_line_args.config
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, #nn information from Neat configuration file
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        conf_name)

strat = cma_strat(44, conf)
nn = strat.show()
print(nn._get_total_dim(), nn.get_output(np.zeros(strat.n_in, dtype="float32")))


"""
from pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.recurrent_net import RecurrentNet

def make_net(genome, config, batch_size):
    return RecurrentNet.create(genome, config, batch_size)

class neat_nn(policy_nn):

    def __init__(self, genome, config):
        self.config  = config
        self.genome = genome
        self.nn = RecurrentNet.create(genome, config, batch_size)

    def get_output(self, input_):
        return self.nn.activate(input_).numpy()

class neat_strat(optimization_strat):

    strat_name = "hyperneat"
    es = None
    _solutions = []
    _solution_idx = 0

    def __init__(self, seed, config, name):
        self.name = name
        self.config = config
        self.p = neat.Population(config)
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)

        self.p.add_reporter(neat.Checkpointer(CHECKPOINT_GENERATION_INTERVAL, filename_prefix=CHECKPOINT_PREFIX))

        # Add a stdout reporter to show progress in the terminal.
        self.p.add_reporter(neat.StdOutReporter(False))

        self.winner = self.p.population.values()

        self.p.run = types.MethodType(run, self.p)
    def show(self) -> policy_nn:
        if self._solution_idx >= len(self._solutions): #Update the population
            if self._solution_idx != 0:
                self.p.run(n=1) # run NEAT for a single iteration.
            self._solutions = list(self.p.population.values())
            self._f_values = []
            self._solution_idx = 0
        res = self._solutions[self._solution_idx]
        return neat_nn(res, self.config)

    def tell(self, f: float) -> None:
        self._solutions[self._solution_idx].fitness = f #Update fitness value 
        self._solution_idx += 1


def make_env():
    return gym.make("CartPole-v0")

evaluator = MultiEnvEvaluator(
    make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps, batch_size=batch_size,
)

fitness = evaluator.eval_genome(genome)
from interfaces import *
import neat
from functools import partial
import types

CHECKPOINT_GENERATION_INTERVAL = 1
CHECKPOINT_PREFIX = None

def _eval_genomes(eval_single_genome, genomes, neat_config):
    parallel_evaluator = ParallelEvaluator(NUM_WORKERS, eval_function=eval_single_genome)

    parallel_evaluator.evaluate(genomes, neat_config)

class neat_nn(policy_nn):

    def __init__(self, genome, config):
        self.config  = config
        self.genome = genome
        self.nn = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_output(self, input_):
        
        return self.nn.activate(input_)

def run(self, n):
    #Overidding NEAT package run algorithm so accepts our arguments
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            #self.reporters.start_generation(self.generation)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            #self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness: #Maximization
                self.best_genome = best

            #if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
             #   fv = np.mean([g.fitness for g in self.population.values()])
              #  if fv >= self.config.fitness_threshold:
               #     self.reporters.found_solution(self.config, self.generation, best)
                #    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            #self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        #if self.config.no_fitness_termination:
         #   self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome

class neat_strat(optimization_strat):

    strat_name = "neat"
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

# nn = cma_nn(2,4)
# print(nn._get_total_dim(), nn.get_output(np.array([0.5,0.5])))
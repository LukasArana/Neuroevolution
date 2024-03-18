import numpy as np
import numpy.typing as npt
import csv
import os
import pickle
from pathlib import Path
import pandas as pd

"""
def get_attr(genome):
    #Get attributes from the genome of type cma
    #Number of weights mid
    def get_weights(genome):
        return len(genome.connections)

    def get_neurons(genome):
        return len(genome.nodes)# One node is always the output

    #The activation fucntions
    def get_activations(genome):
        return list([i.activation for i in genome.nodes.values()])
    def get_fitness(genome):
        return genome.fitness
    return [get_weights(genome), get_neurons(genome), get_activations(genome), get_fitness(genome)]
def save_arch(path):
    objects = []
    with open(path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
                objects.append(obj)
            except EOFError:
                break

    data = {"weight":[], "neurons":[], "fitness":[]}
    for obj in objects:
        n_weights, n_neuron, n_activations, fit =  get_attr(obj.genome)

        data["weight"].append(n_weights)
        data["neurons"].append(n_neuron)
        data["fitness"].append(fit)
    name = os.path.splitext(path)[0] + ".csv"
    pd.DataFrame(data).to_csv(name)
"""
class policy_nn:

    def __init__(self, n_in, n_out):
        raise NotImplementedError()

    def get_output(input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: # -1 <= output <= 1
        raise NotImplementedError()

class optimization_strat:
    
    def __init__(self,seed: int, config: str, name: str):
        raise NotImplementedError()

    def show(self) -> policy_nn:
        raise NotImplementedError()

    def tell(self, objective_value: float) -> None:
        raise NotImplementedError()
    def log(self, resfilepath, f, nn, evaluations, steps, time):
        file_exists = os.path.exists(resfilepath)

        directory_path = os.path.dirname(resfilepath)
        # Create all directories in the directory path that do not exist
        os.makedirs(directory_path, exist_ok=True)

        with open(resfilepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['f', 'evaluations', 'steps', 'time'])
            csvwriter.writerow([f, evaluations, steps, time])

        resfilepath = resfilepath.rsplit(".", 1)[0]
        pickle_path = f"{resfilepath}_nn.pkl"
        with open(pickle_path, "ab") as f:
            pickle.dump(nn, f)
        if "neat" in resfilepath:
            save_arch(pickle_path)
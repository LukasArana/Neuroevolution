import pickle
import gymnasium as gym
import graphviz
import pandas as pd
import os
import glob
from cma_strat import cma_nn, cma_strat
from main import evaluate_policy
#from cma_strat import cma_nn 
import numpy as np


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)
    return dot

ENVS = {"cart": gym.make("CartPole-v1", render_mode = "rgb_array"),
        "pendulum": gym.make('Pendulum-v1', render_mode = "rgb_array"),
        "mountain_car_cont": gym.make('MountainCarContinuous-v0', render_mode = "rgb_array"),
        "mountain_car": gym.make("MountainCar-v0", render_mode = "rgb_array"),
        "lunar": gym.make( "LunarLander-v2", render_mode = "rgb_array"),
        "acrobot": gym.make("Acrobot-v1", render_mode = "rgb_array"),
        "DoubleInvertedPendulum": gym.make('InvertedDoublePendulum-v4', render_mode = "rgb_array"),
        "InvertedPendulum": gym.make("InvertedPendulum-v4", render_mode = "rgb_array")}

def get_nns(path):
    nns_cma = {i :[] for i in range(20)}
    nns_neat = {i :[] for i in range(20)}
    idx_cma = 0
    idx_neat = 0
    for idx, i in enumerate(glob.glob(f"{path}/*.pkl")):
        objects = []
        with open(i, "rb") as f:
            while True:
                try:
                    obj = pickle.load(f)
                    objects.append(obj)
                except EOFError:
                    break
        if "cma" in i:
            nns_cma[idx_cma] = objects
            idx_cma += 1

        else:
            nns_neat[idx_neat] = objects
            idx_neat += 1

    for i in range(3):
        path, _ = os.path.split(path)

    return nns_cma, nns_neat

def diff():
    env = "lunar"
    path = f"/home/walle/Desktop/TFG/nofn/results/data/pruebaF/{env}"
    cma, neat = get_nns(path)
    neat = neat[19]
    SEED=3
    rs = np.random.RandomState(seed=SEED)
    diffs = []
    env = ENVS[env]
    for nn in neat:
        diff = []
        mean, var = evaluate_policy(nn, 100, rs.randint(int(1e8)), env, var = True)
        diffs.append(var)
    
    print(np.mean(diffs))
def save_video(path, n = [19]):
    seed = 2
    name = os.path.basename(path)
    env = ENVS[name]

    nns_cma = get_nns(path)
    nns_neat = get_nns(path)

    folder_cma = os.path.join(path, "video", name, "cma")
    folder_neat = os.path.join(path, "video",name, "neat")

    os.makedirs(folder_cma, exist_ok= True)
    os.makedirs(folder_neat, exist_ok= True)

    for key, val in nns_cma.items():
        if key in n:
            for idx, i in enumerate(val):
                print(idx)
                idx = idx * 1000
                name = os.path.join(folder_cma, f"cma_{seed}_{key}_{idx}_video.avi")
                f = evaluate_policy(i, 1, 2, env, True, name)

    for key, val in nns_neanamet.items():
        if key in n:
            for idx, i in enumerate(val):
                idx = idx * 1000
                name = os.path.join(folder_neat, f"neat_{seed}_{key}_{idx}_video.avi")
                f = evaluate_policy(i, 1, 2, env, True, name)

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

#save_arch(path = "/home/walle/Desktop/TFG/nofn/results/data/prueba/mountain_car_cont/neat_mountain_car_cont_3_9_nn.pkl")
f#or env in ENVS.keys():
  #  save_video(path = os.path.join("/home/walle/Desktop/TFG/nofn/results/data/prueba", env))
#save_video(path = "/home/walle/Desktop/TFG/nofn/results/data/prueba/DoubleInvertedPendulum", n = [19])
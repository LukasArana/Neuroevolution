import pickle
import gymnasium as gym
import graphviz
import pandas as pd
import os

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

ENVS = {"pendulum": gym.make('Pendulum-v1'),
            "mountain_car_cont": gym.make('MountainCarContinuous-v0'),
            "mountain_car": gym.make("MountainCar-v0"),
            "lunar": gym.make( "LunarLander-v2"),
            "cart": gym.make("CartPole-v1"),
            "acrobot": gym.make("Acrobot-v1")}

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

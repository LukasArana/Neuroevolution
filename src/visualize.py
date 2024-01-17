import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


results_cma = "/home/walle/Desktop/TFG/nofn/results/data/cma_cartpole_2.txt"
results_neat = "/home/walle/Desktop/TFG/nofn/results/data/neat_cartpole_2.txt"
results_fs_neat = f"{results_neat[:-4]}_fs.txt"
results_fs_cma = f"{results_cma[:-4]}_fs.txt"

filepath = results_cma
filename = filepath.split('/')[-1]
alg, envname, _ = filename.split('_')

directory = os.path.abspath(os.path.join(filepath,"../.."))


print(f"ENV: {envname} \n ALG: {alg}") 

def read_fs(data):
    fs = []
    with open(data, "r") as f:
        for i in f.readlines():
            i = i.rstrip().split(" ")
            fs.append(list(map(float, i)))
    return fs

cma_fs = read_fs(results_fs_cma)
neat_fs = read_fs(results_fs_neat)

data = pd.read_csv(results_neat)
data = pd.DataFrame({"evaluations": data.evaluations, "neat": neat_fs, "cma": cma_fs})

fig, ax = plt.subplots(figsize=(8, 6))
ax.boxplot(x = data[alg])
ax.set_xticklabels(data["evaluations"])

ax.set_title(f'{envname}', fontsize = 20)
ax.set_xlabel("evaluations", fontsize = 15)
ax.set_ylabel(f"{alg}", fontsize = 15)


figure_name = os.path.join(directory, "figures", f"Box_{alg}_{envname}_{list(data.evaluations)[-1]}.png")
plt.savefig(figure_name, bbox_inches="tight")
plt.show()
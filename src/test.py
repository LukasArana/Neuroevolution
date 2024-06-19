from scipy.stats import mannwhitneyu
import numpy as np
import os
import pandas as pd
import glob
folder=  "results/data/pruebaRandom"
prueba = "DoubleInvertedPendulum"

n = 20 # Number of repetitions
cma = np.zeros(n)
neat = np.zeros(n)
print(f"{os.path.join(folder, prueba)}/*.txt")
for csv_name in glob.glob(f"{os.path.join(folder, prueba)}/*.txt"):
    number = int(os.path.basename(csv_name).split("_")[-1][:-4])
    csv = pd.read_csv(csv_name)
    f = list(csv["f"])[-1]
    if "cmaFC" in csv_name:
        cma[number] = f
    elif "newCMA" in csv_name:
        neat[number] = f

print(len(cma))
print(cma)
print(neat)
print(len(neat))
percentiles = [20, 50, 80]
algs = ["cma", "newCMA"]
for alg in algs:
    for i in percentiles:
        a = cma if alg == "cma" else neat
        print(f"{alg} {i} = {round(np.percentile(a, i), 2)}")
print(mannwhitneyu(cma , neat, alternative='two-sided'))

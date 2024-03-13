from scipy.stats import mannwhitneyu
import numpy as np
import os
import pandas as pd
import glob
folder2=  "/home/walle/Desktop/TFG/nofn/results/data/prueba"
folder = "/home/walle/Desktop/TFG/nofn/results/data/pruebaF"
folders = [folder2]
prueba = "pendulum"
n = 20 # Number of repetitions
cma = np.zeros(n)
neat = np.zeros(n)
for idx, i in enumerate(folders):
    for csv_name in glob.glob(f"{os.path.join(i, prueba)}/*.txt"):
        number = idx * int(n/len(folders)) + int(os.path.basename(csv_name).split("_")[-1][:-4])
        csv = pd.read_csv(csv_name)
        f = list(csv["f"])[-1]
        if "cma" in csv_name:
            cma[number] = f
            #cma[number] = 
        elif "neat" in csv_name:
            neat[number] = f

print(cma)
print(neat)
print(np.median(cma))
print(sum(cma > neat))
print(mannwhitneyu(cma , neat, alternative='two-sided'))

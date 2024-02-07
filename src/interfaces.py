import numpy as np
import numpy.typing as npt
import csv
import os
import pickle
from pathlib import Path
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
    def log(self, resfilepath, f, evaluations, steps, time):
        file_exists = os.path.exists(resfilepath)
        with open(resfilepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if not file_exists:
                csvwriter.writerow(['f', 'evaluations', 'steps', 'time'])
            csvwriter.writerow([f, evaluations, steps, time])
        
        """
        fs_name = f"{resfilepath[:-4]}_fs.txt"
        with open(fs_name, "a") as file:
            for f in fs:
                file.write(str(f) + ' ')
            file.write("\n")
        """
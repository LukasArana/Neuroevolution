import numpy as np
import numpy.typing as npt
import csv
import os
import pickle
from nn_analize import save_arch
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
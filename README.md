

### Install

```bash

sudo apt-get install python3-pip
python3 -m pip install --user virtualenv

virtualenv venv

source venv/bin/activate # Run on every terminal launch

pip install -r requirements.txt
```


Demo is available in the file `src/demo.py`. The demo optimizes a forward pass nn of fixed size with cma-es. cma_strat implements the classes in interfaces.py with CMA-ES on a fixed network. 

TODO:
- [*] Add another file neuroevolution_strat.py that also implements the methods described on interfaces.py, but uses neuroevolution instead.
- [*] Add more RL frameworks besides cartpole (take into account Discrete/Box actions and observations).
- [ ] Normalize inputs / outputs in environments.
- [ ] Add source to plot the results.


To run the demo:

```bash
source venv/bin/activate
python src/demo.py --config path to the NEAT config file from configs
````












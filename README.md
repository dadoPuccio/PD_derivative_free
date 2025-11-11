# Penalty Decomposition Derivative-Free Method

Implementation of the Penalty-Decomposition Derivative-Free (```PDDF```) method introduced in 

[Cecere F., Lapucci M., Pucci D. and Sciandrone M. - Penalty decomposition derivative free method for  the minimization of partially separable functions over a convex feasible set - arXiv pre-print (2025)](https://arxiv.org/abs/2503.21631)


## Installation
In order to use ```PDDF``` you will need a working [conda](https://docs.conda.io/projects/conda/en/latest/index.html) installation. We suggest the creation of a new conda environment with the latest version of ```Python```. The required dependencies ```numpy``` and ```PyNomadBBO``` can be installed through:
```
pip install numpy
pip install PyNomadBBO
```
Another required dependency is [PyCUTEst](https://jfowkes.github.io/pycutest/_build/html/index.html), that can be installed following the [installation guide](https://jfowkes.github.io/pycutest/_build/html/install.html).
To run the Lockwood problem make sure to download the source code from [Robert Gramacy's Surrogates Website](https://bobby.gramacy.com/surrogates/) and compile it.

## Usage
We provide the implementation of ```PDDF```, its parallel version ```PDDF_P``` and the implementation of the competitors considered in the paper ```LS``` and ```SALS```. Installing  ```PyNomadBBO``` package it is possible to run also ```MADS```.  

In order to run the experiments, execute the following:
```
python main.py [options]
```

The following arguments shall be specified:

<div align='center'>
  
| Short Option  | Long Option           | Type    | Description                                          | Default           |
|---------------|-----------------------|---------|------------------------------------------------------|-------------------|
| `-p`          | `--problem_class`     | `str`   | Class of problems to be executed                     | None (required)   |
| `-a`          | `--algorithm`         | `str`   | Algorithm to be executed                             | None (required)   |
| `-ld`         | `--logs_dir`          | `str`   | Path to save the output logs                         | None (required)   |
| `-pause`      | `--pause`             | `float`   | Pause (in seconds) at each function evaluation     | `0.`    |
| `-toll`       | `--toll`              | `float`   | Termination tolerance                              | `1e-4`  |
| `-tl`         | `--time_limt`          | `int`   | Time limit (in seconds)                               | `600`   |
| `-mg`         | `--max_eval_group_k`   | `int`   | Max number of evaulations (normalized by `n` and `m` for data profiles) | `1000`   |

</div>

In `main.py`, the complete list of algorithm parameters can be found and adjusted according to the specific problem considered. 

It is possible to run the experiments reported in the paper with
```
python main.py -p UNCONSTR -a STD -ld output_dir
python main.py -p BOUNDED -a STD -ld output_dir
python main.py -p LOCKWOOD -a STD -ld output_dir -toll 0.01 -tl 3600

python main.py -p UNCONSTR -a FAST -ld output_dir -pause 0.001 -tl 3600
python main.py -p BOUNDED -a FAST -ld output_dir -pause 0.001 -tl 3600
```


## Credits
In case you employed our code for research purposes, please cite:

```
@misc{cecere2025penaltydecompositionderivativefree,
      title={Penalty decomposition derivative free method for the minimization of partially separable functions over a convex feasible set}, 
      author={Francesco Cecere and Matteo Lapucci and Davide Pucci and Marco Sciandrone},
      year={2025},
      eprint={2503.21631},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2503.21631}, 
}
```

# Penalty Decomposition Derivative-Free Method

Implementation of the Penalty-Decomposition Derivative-Free (```PDDF```) method introduced in 

[Cecere F., Lapucci M., Pucci D. and Sciandrone M. - Penalty decomposition derivative free method for  the minimization of partially separable functions over a convex feasible set - arXiv pre-print (2025)](https://arxiv.org/abs/2503.21631)

## Installation
In order to use ```PDDF``` you will need a working [conda](https://docs.conda.io/projects/conda/en/latest/index.html) installation. We suggest the creation of a new conda environment with the latest version of ```Python```. The required dependencies ```numpy``` and ```dill``` can be installed through:
```
pip install numpy
pip install dill
```

## Usage
We provide the implementation of both ```PDDF``` and its parallel version. We also deliver an implementation of the problem ```ARWHEAD``` considered in the paper, which can be used to test the algorithm and as an example for implementing other problems within our experimental framwork.

In order to run the example, execute the following:
```
python main.py -p SMALL -ld logs -a PDDF -toll 0.0001 
```
In main.py, the complete list of algorithm parameters can be found and adjusted according to the specific problem considered. 

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

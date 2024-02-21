# Particle Swarm Optimization

This Python class implements the Particle Swarm Optimization (PSO) algorithm for optimizing any given cost function. PSO is a population-based stochastic optimization algorithm inspired by the social behavior of birds flocking or fish schooling.

## Installation
To use this PSO optimizer, simply include the provided ParticleSwarmOptimizer class in your Python project.

## Usage
1. Import the PSO optimizer, follow these steps:
```python
from pso import ParticleSwarmOptimizer as PSO
```
2. Define your cost function. It should take a list of parameters as input and return a single scalar value representing the cost:

```python
def my_cost_function(params:list | tuple) -> float:
    # Your implementation here
    pass
```
3. Initialize the PSO optimizer with appropriate parameters:
```python
pso = PSO(cost_function=my_cost_function, nvars=10, min_vars=[0]*10, max_vars=[1]*10)
```
4. Execute the optimization:
```python
best_solution = pso.executer()
```

## Parameters
- cost_function: Your cost function that you want to minimize (or maximize if maximization is set to True).
- maximization: Boolean indicating whether the optimization is a maximization problem (default is True).
- nvars: The number of variables in the optimization problem.
- min_vars, max_vars: Lists defining the lower and upper bounds for each variable.
- npop: The number of particles in the swarm.
- interation_limit: The maximum number of iterations for optimization.
- kappa: A parameter for calculating the constriction coefficient.
- phis: A list containing the cognitive and social parameters.
- wdamp: Damping factor for the inertia weight.
- info_display: Boolean indicating whether to display logging information (default is False).
- output_path: Path to save output files such as plots and CSVs.

## Output
The optimizer provides various output methods such as saving particle positions and costs to a CSV file and plotting the convergence behavior.

## Example
Here's a simple example demonstrating the usage of the PSO optimizer:
```python
from particle_swarm_optimizer import ParticleSwarmOptimizer as PSO

def sphere_function(params):
    return sum(p**2 for p in params)

pso = PSO(cost_function=sphere_function, nvars=3, min_vars=[-5]*3, max_vars=[5]*3)
best_solution = pso.executer()
```
## License
This PSO optimizer is licensed under the MIT License. See the LICENSE file for details.


import py_attractor.model as model
import py_attractor.cool_attractors as cool_attractors
import numpy as np
import math

def run_model(model: model.model, n: int):
    results = np.zeros((3,n))
    for i in range(n):
        model.process_step()
        results[0,i], results[1,i], results[2,i] = model.get_state()
    
    return results

def reduce_solution_size(result: np.array, jump: int):
    n_vals = math.floor(result.shape[1]/jump)

    new_results = np.zeros((result.shape[0], n_vals))
    for i in range(n_vals):
        new_results[:, i] = result[:, i*jump]
    
    return new_results

def main():
    m = cool_attractors.aizawaAttractor([0.1, 0, 0], 0.000001, 0.95, 0.7, 0.65, 3.5, 0.25, 0.1)

    _ = run_model(m, 100000)

if __name__ == "__main__":
    main()
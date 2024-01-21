import py_attractor.cool_attractors as cool_attractors
import py_attractor.evaluate as evaluate

import matplotlib.pyplot as plt

def main():
    # m = cool_attractors.aizawaAttractor(
    #     initial_state=[0.1, 0, 0], 
    #     dt=0.00001, 
    #     a=1, 
    #     b=0.7, 
    #     c=0.6, 
    #     d=3.5, 
    #     e=0.25,
    #     f=0.1)
    
    m = cool_attractors.arneodoAttractor(
        initial_state=[0.1, 0, 0],
        dt=0.00005,
        a=-5.5,
        b=3.5,
        c=-1.0)

    results = evaluate.run_model(m, 5000000)
    results = evaluate.reduce_solution_size(results, 50)

    axs = plt.figure().add_subplot(projection='3d')
    axs.plot(results[0,:], results[1,:], results[2,:])
    plt.show()

if __name__ == "__main__":
    main()
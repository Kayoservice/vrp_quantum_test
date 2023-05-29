import os

import matplotlib.pyplot as plt
import numpy as np
from qiskit_optimization.applications.vehicle_routing import VehicleRouting

from graph import OSMGraph


# Get the data
class VrpSolving:
    def __init__(self, vehicles: int, perimeters: int):
        self.graph, self.depot = OSMGraph(
            perimeters, os.environ["address"]
        ).generate_graph_from_address()
        self.vehicles = vehicles

    def get_vrp_problem(self):
        print("result:")
        print(
            VehicleRouting(self.graph, self.vehicles, self.depot).to_quadratic_program()
        )

    def generate_instance(self):
        n = self.n

        # np.random.seed(33)
        np.random.seed(1543)

        xc = (np.random.rand(n) - 0.5) * 10
        yc = (np.random.rand(n) - 0.5) * 10

        instance = np.zeros([n, n])
        for ii in range(0, n):
            for jj in range(ii + 1, n):
                instance[ii, jj] = (xc[ii] - xc[jj]) ** 2 + (yc[ii] - yc[jj]) ** 2
                instance[jj, ii] = instance[ii, jj]

        return xc, yc, instance


# Visualize the solution
def visualize_solution(xc, yc, x, C, n, title_str):
    plt.figure()
    plt.scatter(xc, yc, s=200)
    for i in range(len(xc)):
        plt.annotate(i, (xc[i] + 0.15, yc[i]), size=16, color="r")
    plt.plot(xc[0], yc[0], "r*", ms=20)

    plt.grid()

    for ii in range(0, n**2):
        if x[ii] > 0:
            ix = ii // n
            iy = ii % n
            plt.arrow(
                xc[ix],
                yc[ix],
                xc[iy] - xc[ix],
                yc[iy] - yc[ix],
                length_includes_head=True,
                head_width=0.25,
            )

    plt.title(title_str + " cost = " + str(int(C * 100) / 100.0))
    plt.savefig(f"test{title_str}.png")
    plt.show()


# OSMGraph(distance_kms=1500,address=os.environ['address']).generate_graph_from_address()
# # Initialize the problem by defining the parameters
# nodes = 3  # number of nodes + depot (n+1)
# vehicles = 2  # number of vehicles


# # Initialize the problem by randomly generating the instance
# initializer = VrpInitializer(nodes)
# xc, yc, instance = initializer.generate_instance()
# print(xc,yc,instance)

# # Instantiate the classical optimizer class
# classical_optimizer = ClassicalOptimizer(instance, nodes, vehicles)

# # Print number of feasible solutions
# print("Number of feasible solutions = " + str(classical_optimizer.compute_allowed_combinations()))

# # Solve the problem in a classical fashion via CPLEX
# x = None
# z = None
# try:
#     x, classical_cost = classical_optimizer.cplex_solution()
#     # Put the solution in the z variable
#     z = [x[ii] for ii in range(nodes**2) if ii // nodes != ii % nodes]
#     # Print the solution
#     print(z)
# except:
#     print("CPLEX may be missing.")

# if x is not None:
#     visualize_solution(xc, yc, x, classical_cost, nodes, "Classical")

# # Instantiate the quantum optimizer class with parameters:
# quantum_optimizer = QuantumOptimizer(instance, nodes, vehicles)

# # Check if the binary representation is correct
# try:
#     if z is not None:
#         Q, g, c, binary_cost = quantum_optimizer.binary_representation(x_sol=z)
#         print("Binary cost:", binary_cost, "classical cost:", classical_cost)
#         if np.abs(binary_cost - classical_cost) < 0.01:
#             print("Binary formulation is correct")
#         else:
#             print("Error in the binary formulation")
#     else:
#         print("Could not verify the correctness, due to CPLEX solution being unavailable.")
#         Q, g, c, binary_cost = quantum_optimizer.binary_representation()
#         print("Binary cost:", binary_cost)
# except NameError as e:
#     print("Warning: Please run the cells above first.")
#     print(e)

# qp = quantum_optimizer.construct_problem(nodes,Q, g, c)
# quantum_solution, quantum_cost = quantum_optimizer.solve_problem(qp)

# print(quantum_solution, quantum_cost)
# # Put the solution in a way that is compatible with the classical variables
# x_quantum = np.zeros(nodes**2)
# kk = 0
# for ii in range(nodes**2):
#     if ii // nodes != ii % nodes:
#         x_quantum[ii] = quantum_solution[kk]
#         kk += 1


# # visualize the solution
# visualize_solution(xc, yc, x_quantum, quantum_cost, nodes, "Quantum")

# # and visualize the classical for comparison
# if x is not None:
#     visualize_solution(xc, yc, x, classical_cost, nodes, "Classical")

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications.vehicle_routing import VehicleRouting


# Get the data
class VrpSolving:
    def __init__(self, vehicles: int, depot: int, nodes: int):
        self.vehicles = vehicles
        self.depot = depot
        self.nodes = nodes

    def solve_vrp_problem(self):
        plt.figure()
        graph: nx.DiGraph = nx.complete_graph(self.nodes, nx.DiGraph())
        for (start, end) in graph.edges:
            graph.edges[start, end]["weight"] = round(random.random())
        nx.draw(graph, with_labels=True, font_weight="bold")
        plt.savefig("graph.png")
        problem_instance = VehicleRouting(graph, num_vehicles=self.vehicles, depot=2)
        problem_program = problem_instance.to_quadratic_program()
        quantum_optimizer = QuantumOptimizer(
            problem_instance, self.nodes, self.vehicles
        )
        print("solving problem....")
        quantum_solution, quantum_cost = quantum_optimizer.solve_problem(
            problem_program
        )
        print(quantum_solution, quantum_cost)


class QuantumOptimizer:
    def __init__(self, instance, n, K):
        self.instance = instance
        self.n = n
        self.K = K

    def binary_representation(self, x_sol=0):
        instance = self.instance
        n = self.n
        K = self.K

        A = np.max(instance) * 100  # A parameter of cost function

        # Determine the weights w
        instance_vec = instance.reshape(n**2)
        w_list = [instance_vec[x] for x in range(n**2) if instance_vec[x] > 0]
        w = np.zeros(n * (n - 1))
        for ii in range(len(w_list)):
            w[ii] = w_list[ii]

        # Some variables I will use
        Id_n = np.eye(n)
        Im_n_1 = np.ones([n - 1, n - 1])
        Iv_n_1 = np.ones(n)
        Iv_n_1[0] = 0
        Iv_n = np.ones(n - 1)
        neg_Iv_n_1 = np.ones(n) - Iv_n_1

        v = np.zeros([n, n * (n - 1)])
        for ii in range(n):
            count = ii - 1
            for jj in range(n * (n - 1)):
                if jj // (n - 1) == ii:
                    count = ii

                if jj // (n - 1) != ii and jj % (n - 1) == count:
                    v[ii][jj] = 1.0

        vn = np.sum(v[1:], axis=0)

        # Q defines the interactions between variables
        Q = A * (np.kron(Id_n, Im_n_1) + np.dot(v.T, v))

        # g defines the contribution from the individual variables
        g = (
            w
            - 2 * A * (np.kron(Iv_n_1, Iv_n) + vn.T)
            - 2 * A * K * (np.kron(neg_Iv_n_1, Iv_n) + v[0].T)
        )

        # c is the constant offset
        c = 2 * A * (n - 1) + 2 * A * (K**2)

        try:
            max(x_sol)
            # Evaluates the cost distance from a binary representation of a path
            fun = (
                lambda x: np.dot(np.around(x), np.dot(Q, np.around(x)))
                + np.dot(g, np.around(x))
                + c
            )
            cost = fun(x_sol)
        except:
            cost = 0

        return Q, g, c, cost

    def solve_problem(self, qp):
        algorithm_globals.random_seed = 10598
        vqe = SamplingVQE(sampler=Sampler(), optimizer=SPSA(), ansatz=RealAmplitudes())
        optimizer = MinimumEigenOptimizer(min_eigen_solver=vqe)
        print("solving using optimizer")
        try:

            result = optimizer.solve(qp)
        except Exception as error:
            raise error

        print("compute result")
        # compute cost of the obtained result
        _, _, _, level = self.binary_representation(x_sol=result.x)
        return result.x, level

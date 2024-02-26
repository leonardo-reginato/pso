import unittest
import os
from pso.optimizer import PSO


class TestParticleSwarmOptimizer(unittest.TestCase):
    def cost_function(self, x, y):
        return x + y

    def test_initialization(self):
        # Test initialization of ParticleSwarmOptimizer
        pso = PSO(
            cost_function=self.cost_function, min_vars=[0, 0], max_vars=[1, 1]
        )

        self.assertIsNotNone(pso)
        self.assertEqual(len(pso.min_vars), 2)
        self.assertEqual(len(pso.max_vars), 2)
        self.assertEqual(pso.npop, 2)

    def test_optimization_min(self):
        # Test optimization with a simple cost function (minimization)
        pso = PSO(
            cost_function=self.cost_function,
            min_vars=[0, 0],
            max_vars=[1, 1],
            npop=5,
            interation_limit=5,
            maximization=False,
        )
        best_solution = pso.executer()

        self.assertIsNotNone(best_solution)
        self.assertLessEqual(best_solution["cost"], 0.5)

    def test_optimization_max(self):
        # Test optimization with a simple cost function (minimization)
        pso = PSO(
            cost_function=self.cost_function,
            min_vars=[0, 0],
            max_vars=[1, 1],
            npop=5,
            interation_limit=5,
            maximization=True,
        )
        best_solution = pso.executer()

        self.assertIsNotNone(best_solution)
        self.assertGreaterEqual(best_solution["cost"], 0.8)

    def test_output(self):
        # Test if output files are saved properly
        pso = PSO(
            cost_function=self.cost_function,
            min_vars=[0, 0],
            max_vars=[1, 1],
            npop=2,
            interation_limit=5,
            maximization=False,
        )

        pso.executer()

        # Assuming output files are saved in the current directory
        self.assertTrue(
            os.path.exists(os.getcwd() + "/output/position_cost_result.csv")
        )
        self.assertTrue(os.path.exists(os.getcwd() + "/output/iteration_plot.png"))

    def test_bounds(self):
        # Test if particle positions are within specified bounds
        pso = PSO(
            cost_function=self.cost_function,
            min_vars=[0, 0],
            max_vars=[1, 1],
            npop=2,
            interation_limit=5,
            maximization=False,
        )

        best_solution = pso.executer()
        positions = best_solution["position"]

        self.assertTrue(all(pos >= 0 for pos in positions))
        self.assertTrue(all(pos <= 1 for pos in positions))

    def test_global_best(self):
        # Test if the global best solution is correctly updated
        pso = PSO(
            cost_function=self.cost_function,
            min_vars=[0, 0],
            max_vars=[1, 1],
            npop=5,
            interation_limit=10,
            maximization=False,
        )

        best_solution = pso.executer()

        # Assuming minimization problem, global best cost should be close to 0
        self.assertLessEqual(best_solution["cost"], 1)


if __name__ == "__main__":
    unittest.main()

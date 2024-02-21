import unittest
import os
from pso import ParticleSwarmOptimizer


class TestParticleSwarmOptimizer(unittest.TestCase):
    def test_initialization(self):
        # Test initialization of ParticleSwarmOptimizer
        def cost_function(x):
            return sum(x)

        pso = ParticleSwarmOptimizer(
            cost_function=cost_function, nvars=2, min_vars=[0, 0], max_vars=[1, 1]
        )

        self.assertIsNotNone(pso)
        self.assertEqual(len(pso.min_vars), 2)
        self.assertEqual(len(pso.max_vars), 2)
        self.assertEqual(pso.npop, 2)

    def test_optimization(self):
        # Test optimization with a simple cost function (minimization)
        cost_function = lambda x: sum(x)
        pso = ParticleSwarmOptimizer(
            cost_function=cost_function, nvars=2, min_vars=[0, 0], max_vars=[1, 1]
        )
        best_solution = pso.executer()

        self.assertIsNotNone(best_solution)
        self.assertAlmostEqual(best_solution["cost"], 0.0, places=5)

    def test_output(self):
        # Test if output files are saved properly
        cost_function = lambda x: sum(x)
        pso = ParticleSwarmOptimizer(
            cost_function=cost_function, nvars=2, min_vars=[0, 0], max_vars=[1, 1]
        )
        best_solution = pso.executer()

        # Assuming output files are saved in the current directory
        self.assertTrue(os.path.exists("position_cost_result.csv"))
        self.assertTrue(os.path.exists("iteration_plot.png"))

    def test_bounds(self):
        # Test if particle positions are within specified bounds
        cost_function = lambda x: sum(x)
        pso = ParticleSwarmOptimizer(
            cost_function=cost_function, nvars=2, min_vars=[0, 0], max_vars=[1, 1]
        )
        best_solution = pso.executer()
        positions = best_solution["position"]

        self.assertTrue(all(pos >= 0 for pos in positions))
        self.assertTrue(all(pos <= 1 for pos in positions))

    def test_global_best(self):
        # Test if the global best solution is correctly updated
        cost_function = lambda x: sum(x)
        pso = ParticleSwarmOptimizer(
            cost_function=cost_function, nvars=2, min_vars=[0, 0], max_vars=[1, 1]
        )
        best_solution = pso.executer()

        # Assuming minimization problem, global best cost should be close to 0
        self.assertAlmostEqual(best_solution["cost"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

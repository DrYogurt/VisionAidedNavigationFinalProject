import unittest

import numpy as np

from src.environment import EnvironmentConfig
from src.experiment import ExperimentRunner


class TestParallelExperiments(unittest.TestCase):
    def test_parallel_and_serial_scientific_outputs_match(self):
        config = EnvironmentConfig(
            num_objects=1,
            num_classes_in_scene=1,
            num_classes_in_model=2,
            num_trials=2,
            num_steps=2,
            num_samples=30,
            sensor_range=100.0,
            max_hypotheses=20,
            sigma_p=np.diag([0.05, 0.05, 0.01]),
        )
        runner = ExperimentRunner(config, seed=17)
        modes = ["viewpoint_dependent", "geometric_only"]
        serial = runner.run_comparative_experiment(modes=modes, workers=1)
        try:
            parallel = runner.run_comparative_experiment(modes=modes, workers=2)
        except PermissionError as exc:
            self.skipTest(f"process creation is blocked by the execution sandbox: {exc}")

        for mode in modes:
            for metric in serial[mode]:
                if metric == "step_time":
                    continue
                np.testing.assert_array_equal(
                    serial[mode][metric], parallel[mode][metric]
                )


if __name__ == "__main__":
    unittest.main()

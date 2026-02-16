from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_PATHS = [
    "config/models.yaml",
    "config/experiments.yaml",
    "config/tasks.yaml",
    "config/azure_foundry.yaml",
    "src/data/generators/base.py",
    "src/models/base.py",
    "src/training/trainer.py",
    "src/evaluation/evaluator.py",
    "src/azure_foundry/client.py",
    "src/orchestration/pipeline.py",
    "src/utils/config/manager.py",
    "scripts/generate_data.py",
    "scripts/run_experiment.py",
    "scripts/evaluate_models.py",
    "scripts/compare_results.py",
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_baseline_evaluation.ipynb",
    "notebooks/03_results_analysis.ipynb",
]


class TestProjectStructure(unittest.TestCase):
    def test_expected_paths_exist(self) -> None:
        for relative_path in EXPECTED_PATHS:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).exists())


if __name__ == "__main__":
    unittest.main()

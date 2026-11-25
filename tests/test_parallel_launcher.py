import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.run_learning_curves import (
    DEFAULT_JOBS_PER_GPU,
    DEFAULT_K,
    REGION_GRAPHS,
    configure_commands,
    run_queue,
    select_commands,
)


class ParallelLauncherTests(unittest.TestCase):
    def test_campaign_defaults_use_three_region_graphs_and_six_slots(self) -> None:
        self.assertEqual(REGION_GRAPHS, ("quad-tree-2", "quad-tree-4", "quad-graph"))
        self.assertNotIn("chow-liu-tree", REGION_GRAPHS)
        self.assertEqual(DEFAULT_JOBS_PER_GPU, 3)
        self.assertEqual(DEFAULT_K, 128)

    def test_configure_commands_overrides_k_and_budgets(self) -> None:
        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_image_circuit",
            "--k",
            "64",
            "--max-epochs",
            "100",
            "--validation-steps",
            "100",
            "--no-early-stopping",
            "--save-path",
            "mnist_data-init_k64_seed42.pt",
        ]
        configured = configure_commands(
            [command],
            k=32,
            max_epochs=3,
            validation_steps=20,
            patience=7,
            progress_steps=5,
            project="test-project",
        )[0]
        self.assertEqual(configured[configured.index("--k") + 1], "32")
        self.assertEqual(configured[configured.index("--max-epochs") + 1], "3")
        self.assertEqual(configured[configured.index("--validation-steps") + 1], "20")
        self.assertEqual(configured[configured.index("--patience") + 1], "7")
        self.assertNotIn("--no-early-stopping", configured)
        self.assertEqual(configured[configured.index("--progress-steps") + 1], "5")
        self.assertEqual(configured[configured.index("--project") + 1], "test-project")
        self.assertIn("_k32_", configured[configured.index("--save-path") + 1])

    def test_selects_one_dataset_and_complete_initialization_pairs(self) -> None:
        def command(dataset: str, rg: str, data_init: bool) -> list[str]:
            result = [
                sys.executable,
                "-m",
                "scripts.evaluate_image_circuit",
                "--dataset",
                dataset,
                "--rg",
                rg,
                "--save-path",
                f"{dataset}_{rg}_{data_init}.pt",
            ]
            if data_init:
                result.append("--use-estimated-weights")
            return result

        matrix = [
            command(dataset, rg, data_init)
            for dataset in ("mnist", "cifar10")
            for rg in ("quad-tree-2", "quad-graph")
            for data_init in (True, False)
        ]
        selected = select_commands(
            matrix, dataset="cifar10", region_graphs=["quad-graph"]
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(
            all(command[command.index("--dataset") + 1] == "cifar10" for command in selected)
        )
        self.assertEqual(
            {"data" if "--use-estimated-weights" in command else "random" for command in selected},
            {"data", "random"},
        )

    def test_project_template_is_resolved_per_dataset_and_rg(self) -> None:
        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_image_circuit",
            "--dataset",
            "fashion-mnist",
            "--rg",
            "quad-tree-4",
            "--save-path",
            "fashion-mnist_quad-tree-4_data-init_k64_seed42.pt",
        ]
        configured = configure_commands(
            [command],
            k=None,
            max_epochs=None,
            validation_steps=None,
            patience=5,
            progress_steps=10,
            project="pc_{dataset}_{rg}_eval",
        )[0]
        self.assertEqual(
            configured[configured.index("--project") + 1],
            "pc_fashion-mnist_quad-tree-4_eval",
        )

    def test_run_queue_reports_process_progress_and_gpu_occupancy(self) -> None:
        child = """\
import time
for step in range(0, 11, 5):
    print(f"EXPERIMENT_PROGRESS step={step} max_steps=10", flush=True)
    time.sleep(0.03)
"""
        commands = [
            [sys.executable, "-c", child, "--save-path", f"fake_{index}.pt"]
            for index in range(4)
        ]
        with tempfile.TemporaryDirectory(prefix="cirkit-dashboard-") as temporary:
            output_dir = Path(temporary)
            code = run_queue(
                commands,
                slots=["0", "0", "1", "1"],
                output_dir=output_dir,
                show_progress=False,
            )
            status = json.loads(
                (output_dir / "run-status.json").read_text(encoding="utf-8")
            )

        self.assertEqual(code, 0)
        self.assertEqual(status["completed"], 4)
        self.assertEqual(status["failed"], 0)
        self.assertEqual(status["queued"], 0)
        self.assertEqual(status["gpus"]["0"]["capacity"], 2)
        self.assertEqual(status["gpus"]["1"]["capacity"], 2)
        self.assertTrue(all(result["step"] == 10 for result in status["results"]))


if __name__ == "__main__":
    unittest.main()

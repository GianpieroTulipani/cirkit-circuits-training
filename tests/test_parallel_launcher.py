import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.run_learning_curves import configure_commands, run_queue


class ParallelLauncherTests(unittest.TestCase):
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
            "--save-path",
            "mnist_data-init_k64_seed42.pt",
        ]
        configured = configure_commands(
            [command],
            k=32,
            max_epochs=3,
            validation_steps=20,
            progress_steps=5,
            project="test-project",
        )[0]
        self.assertEqual(configured[configured.index("--k") + 1], "32")
        self.assertEqual(configured[configured.index("--max-epochs") + 1], "3")
        self.assertEqual(configured[configured.index("--validation-steps") + 1], "20")
        self.assertEqual(configured[configured.index("--progress-steps") + 1], "5")
        self.assertEqual(configured[configured.index("--project") + 1], "test-project")
        self.assertIn("_k32_", configured[configured.index("--save-path") + 1])

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

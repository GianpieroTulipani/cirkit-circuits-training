"""Run notebook experiments concurrently across a configurable set of GPUs."""

import argparse
from collections import Counter, deque
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_RE = re.compile(r"EXPERIMENT_PROGRESS step=(\d+) max_steps=(\d+)")
DATASETS = ("mnist", "fashion-mnist", "cifar10")
REGION_GRAPHS = ("quad-tree-2", "quad-tree-4", "quad-graph")
DEFAULT_JOBS_PER_GPU = 3
DEFAULT_K = 128


def notebook_commands(notebook: Path) -> list[list[str]]:
    """Reuse the notebook's experiment settings without executing notebook cells."""
    commands = []
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or not source.startswith(
            "!python -m scripts.evaluate_image_circuit"
        ):
            continue
        command = shlex.split(source.replace("\\\n", " "))
        commands.append([sys.executable, *command[1:]])
    if not commands:
        raise ValueError(f"No training commands found in {notebook}")
    return commands


def _set_option(command: list[str], option: str, value: object) -> None:
    """Replace a CLI option or append it when the notebook command omits it."""
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend((option, str(value)))


def _remove_flag(command: list[str], flag: str) -> None:
    """Remove every occurrence of a boolean CLI flag."""
    while flag in command:
        command.remove(flag)


def _checkpoint_name(command: list[str]) -> str:
    return Path(command[command.index("--save-path") + 1]).stem


def _option_value(command: list[str], option: str) -> str:
    """Return a required option value from a generated training command."""
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Generated command is missing {option}: {shlex.join(command)}") from error


def select_commands(
    commands: list[list[str]], *, dataset: str, region_graphs: list[str]
) -> list[list[str]]:
    """Select complete data-init/random-init comparisons for one dataset."""
    selected = []
    for rg in region_graphs:
        selected.extend(
            command
            for command in commands
            if _option_value(command, "--dataset") == dataset
            and _option_value(command, "--rg") == rg
        )
    expected = 2 * len(region_graphs)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} experiments for dataset={dataset} and RGs={region_graphs}, "
            f"but found {len(selected)}"
        )
    for rg in region_graphs:
        pair = [command for command in selected if _option_value(command, "--rg") == rg]
        methods = {
            "data-init" if "--use-estimated-weights" in command else "random-init"
            for command in pair
        }
        if methods != {"data-init", "random-init"}:
            raise ValueError(f"Incomplete initialization pair for dataset={dataset}, rg={rg}")
    return selected


def _format_project(template: str, command: list[str]) -> str:
    """Build the W&B project name for a dataset/RG comparison."""
    if "{" not in template:
        return template
    values = {
        "dataset": _option_value(command, "--dataset"),
        "rg": _option_value(command, "--rg"),
    }
    try:
        return template.format(**values)
    except KeyError as error:
        raise ValueError(
            f"Unknown W&B project-template field {error.args[0]!r}; use {{dataset}} and {{rg}}"
        ) from error


def configure_commands(
    commands: list[list[str]],
    *,
    k: int | None,
    max_epochs: int | None,
    validation_steps: int | None,
    patience: int,
    progress_steps: int,
    project: str | None,
) -> list[list[str]]:
    """Apply launcher-level overrides while keeping the experiment matrix in one place."""
    configured = []
    for original in commands:
        command = original.copy()
        if k is not None:
            _set_option(command, "--k", k)
            save_index = command.index("--save-path") + 1
            save_path = Path(command[save_index])
            stem = re.sub(r"_k\d+(?=_|$)", f"_k{k}", save_path.stem)
            if stem == save_path.stem and f"_k{k}" not in stem:
                stem = f"{stem}_k{k}"
            command[save_index] = str(save_path.with_name(stem + save_path.suffix))
        if max_epochs is not None:
            _set_option(command, "--max-epochs", max_epochs)
        if validation_steps is not None:
            _set_option(command, "--validation-steps", validation_steps)
        # Campaigns always keep early stopping active, even if an older source
        # notebook command explicitly disabled it.
        _remove_flag(command, "--no-early-stopping")
        _set_option(command, "--patience", patience)
        _set_option(command, "--progress-steps", progress_steps)
        if project is not None:
            _set_option(command, "--project", _format_project(project, command))
        configured.append(command)
    return configured


def _read_progress(run: dict) -> bool:
    """Read newly appended log data and update a run's optimizer-step progress."""
    try:
        with run["log_path"].open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(run["read_offset"])
            chunk = stream.read()
            run["read_offset"] = stream.tell()
    except FileNotFoundError:
        return False
    if not chunk:
        return False
    data = run["tail"] + chunk
    matches = list(PROGRESS_RE.finditer(data))
    run["tail"] = data[-256:]
    if not matches:
        return False
    match = matches[-1]
    run["step"] = int(match.group(1))
    run["max_steps"] = int(match.group(2))
    return True


def _status_payload(
    pending: deque[list[str]], active: dict[int, dict], results: list[dict], slots: list[str]
) -> dict:
    gpu_capacity = Counter(slots)
    gpu_active = Counter(run["gpu"] for run in active.values())
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "queued": len(pending),
        "completed": sum(result["status"] == "done" for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "gpus": {
            gpu: {"active": gpu_active[gpu], "capacity": gpu_capacity[gpu]}
            for gpu in gpu_capacity
        },
        "active": [
            {
                "name": run["name"],
                "gpu": run["gpu"],
                "slot": slot,
                "step": run["step"],
                "max_steps": run["max_steps"],
                "elapsed_seconds": round(time.monotonic() - run["started_monotonic"], 1),
                "log": str(run["log_path"]),
            }
            for slot, run in sorted(active.items())
        ],
        "results": results,
    }


def run_queue(
    commands: list[list[str]], slots: list[str], output_dir: Path, show_progress: bool
) -> int:
    """Refill GPU slots and show only the experiments currently occupying a GPU."""
    pending = deque(commands)
    active: dict[int, dict] = {}
    results: list[dict] = []
    failures = 0
    status_path = output_dir / "run-status.json"
    bars = {}
    overall = None

    if show_progress:
        from tqdm.auto import tqdm

        overall = tqdm(
            total=len(commands), desc="All experiments", position=0, dynamic_ncols=True
        )
        gpu_slot_number = Counter()
        for slot, gpu in enumerate(slots):
            gpu_slot_number[gpu] += 1
            bars[slot] = tqdm(
                total=100,
                desc=f"GPU {gpu} slot {gpu_slot_number[gpu]}: idle",
                position=slot + 1,
                leave=True,
                dynamic_ncols=True,
            )

    def announce(message: str) -> None:
        # A live dashboard must not grow by one line whenever a process starts or
        # finishes. Plain status messages are kept for redirected/headless logs.
        if not show_progress:
            print(message, flush=True)

    def refresh_dashboard() -> None:
        for slot, run in active.items():
            changed = _read_progress(run)
            if show_progress and (changed or run["max_steps"] is None):
                bar = bars[slot]
                if run["max_steps"] is not None:
                    bar.total = run["max_steps"]
                    bar.n = min(run["step"], run["max_steps"])
                elapsed = time.monotonic() - run["started_monotonic"]
                bar.set_postfix_str(f"{elapsed / 60:.1f}m")
                bar.refresh()
        payload = _status_payload(pending, active, results, slots)
        temporary_status = status_path.with_suffix(".json.tmp")
        temporary_status.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary_status.replace(status_path)
        if overall is not None:
            gpu_capacity = Counter(slots)
            gpu_active = Counter(run["gpu"] for run in active.values())
            occupancy = " ".join(
                f"GPU{gpu}={gpu_active[gpu]}/{gpu_capacity[gpu]}"
                for gpu in gpu_capacity
            )
            overall.set_postfix_str(
                f"{occupancy} queued={len(pending)} failed={failures}"
            )

    try:
        while pending or active:
            for slot, gpu in enumerate(slots):
                if slot in active or not pending:
                    continue
                command = pending.popleft().copy()
                save_index = command.index("--save-path") + 1
                name = Path(command[save_index]).stem
                checkpoint_path = output_dir / f"{name}.pt"
                command[save_index] = str(checkpoint_path)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                env["PYTHONUNBUFFERED"] = "1"
                # Kaggle's CPU cores are shared by all concurrent experiments.
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                log_path = output_dir / f"{name}.log"
                log = log_path.open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                active[slot] = {
                    "process": process,
                    "log": log,
                    "log_path": log_path,
                    "checkpoint": checkpoint_path,
                    "name": name,
                    "gpu": gpu,
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "started_monotonic": time.monotonic(),
                    "read_offset": 0,
                    "tail": "",
                    "step": 0,
                    "max_steps": None,
                }
                if show_progress:
                    bar = bars[slot]
                    bar.reset(total=100)
                    bar.set_description_str(f"GPU {gpu}: {name}")
                    bar.set_postfix_str("starting")
                    bar.refresh()
                announce(f"START GPU {gpu}, slot {slot}: {name}")

            refresh_dashboard()

            for slot, run in list(active.items()):
                code = run["process"].poll()
                if code is None:
                    continue
                _read_progress(run)
                run["log"].close()
                del active[slot]
                status = "done" if code == 0 else "failed"
                failures += code != 0
                elapsed = time.monotonic() - run["started_monotonic"]
                results.append(
                    {
                        "name": run["name"],
                        "gpu": run["gpu"],
                        "slot": slot,
                        "status": status,
                        "return_code": code,
                        "step": run["step"],
                        "max_steps": run["max_steps"],
                        "started_at": run["started_at"],
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                        "elapsed_seconds": round(elapsed, 1),
                        "log": str(run["log_path"]),
                        "checkpoint": str(run["checkpoint"]),
                    }
                )
                if show_progress:
                    bar = bars[slot]
                    if code == 0:
                        bar.total = run["max_steps"] or 1
                        bar.n = bar.total
                    bar.set_description_str(
                        f"GPU {run['gpu']}: {run['name']} [{status.upper()}]"
                    )
                    bar.set_postfix_str(f"{elapsed / 60:.1f}m")
                    bar.refresh()
                    overall.update(1)
                announce(f"{status.upper()} ({code}): {run['name']}")

            refresh_dashboard()
            if active:
                time.sleep(1)
    finally:
        # Interrupting the launcher also stops all of its training processes.
        for run in active.values():
            if run["process"].poll() is None:
                run["process"].terminate()
        for run in active.values():
            try:
                run["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                run["process"].kill()
                run["process"].wait()
            run["log"].close()
        if overall is not None:
            overall.close()
        for bar in bars.values():
            bar.close()
    return int(failures > 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--jobs-per-gpu", type=int, default=DEFAULT_JOBS_PER_GPU)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="mnist",
        help="Run only this dataset; every selected RG gets a data-init/random-init pair.",
    )
    parser.add_argument(
        "--region-graphs",
        nargs="+",
        choices=REGION_GRAPHS,
        default=list(REGION_GRAPHS),
        help="Region graphs to compare for the selected dataset.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        help="Optional even debug cap; full campaigns should omit it.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of units for every selected experiment.",
    )
    parser.add_argument("--max-epochs", type=int, help="Override the notebook epoch budget.")
    parser.add_argument("--validation-steps", type=int)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stop after this many validation checks without improvement.",
    )
    parser.add_argument("--progress-steps", type=int, default=10)
    project_group = parser.add_mutually_exclusive_group()
    project_group.add_argument(
        "--project-template",
        default="pc_{dataset}_{rg}_eval",
        help="W&B project template; supports {dataset} and {rg}.",
    )
    project_group.add_argument(
        "--project",
        help="Legacy static W&B project override; puts every selected comparison together.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=PROJECT_ROOT / "notebooks" / "data-driven-initialization.ipynb",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Use this exact campaign directory instead of creating one under --output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="With --dry-run, print every generated experiment command.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    for option in (
        "jobs_per_gpu",
        "max_experiments",
        "k",
        "max_epochs",
        "validation_steps",
        "patience",
        "progress_steps",
    ):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain distinct GPU identifiers")
    if args.max_experiments is not None and args.max_experiments % 2:
        parser.error("--max-experiments must be even to preserve initialization pairs")

    commands = select_commands(
        notebook_commands(args.notebook),
        dataset=args.dataset,
        region_graphs=args.region_graphs,
    )
    if args.max_experiments is not None:
        if args.max_experiments > len(commands):
            parser.error(
                f"--max-experiments={args.max_experiments}, but the matrix has "
                f"only {len(commands)} experiments"
            )
        commands = commands[: args.max_experiments]
    commands = configure_commands(
        commands,
        k=args.k,
        max_epochs=args.max_epochs,
        validation_steps=args.validation_steps,
        patience=args.patience,
        progress_steps=args.progress_steps,
        project=args.project or args.project_template,
    )

    slots = [gpu for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    names = [_checkpoint_name(command) for command in commands]
    if len(names) != len(set(names)):
        parser.error("Experiment checkpoint names must be unique")
    print(
        f"Dataset {args.dataset}; RGs {args.region_graphs}; "
        f"{len(commands)} experiments; {len(slots)} process slots across GPUs {args.gpus}"
    )
    projects = sorted({_option_value(command, "--project") for command in commands})
    print("W&B projects: " + ", ".join(projects))
    print(f"Early stopping: enabled; patience={args.patience} validation checks")
    print(
        "Physical GPUs: "
        f"{len(args.gpus)}; concurrent experiments: {len(slots)} "
        f"({args.jobs_per_gpu} per GPU)"
    )
    if args.dry_run:
        print(
            f"Queue: {len(commands)} experiments; initial occupancy: "
            + ", ".join(
                f"GPU {gpu}={args.jobs_per_gpu}" for gpu in args.gpus
            )
        )
        if args.list_commands:
            for command in commands:
                print(shlex.join(command))
        return 0

    import torch
    from scripts.evaluate_image_circuit import _load_vision_dataset

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = (
        visible.split(",")
        if visible is not None
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    if not torch.cuda.is_available() or not set(args.gpus).issubset(available):
        parser.error(f"Requested GPUs {args.gpus}; available GPU identifiers: {available}")
    for gpu in args.gpus:
        print(f"GPU {gpu}: {torch.cuda.get_device_name(int(gpu))}")
    if args.jobs_per_gpu > 2:
        print(
            "WARNING: more than two experiments share each GPU. Reduce jobs-per-gpu "
            "if CUDA runs out of memory or throughput becomes worse.",
            flush=True,
        )

    output_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else args.output_dir.resolve()
        / datetime.now().strftime("curves-%Y%m%d-%H%M%S-%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Logs, checkpoints and run-status.json: {output_dir}", flush=True)
    # Download sequentially before workers start to avoid concurrent extraction.
    datasets = sorted({cmd[cmd.index("--dataset") + 1] for cmd in commands})
    for dataset in datasets:
        print(f"Preparing {dataset}", flush=True)
        _load_vision_dataset(dataset, str(PROJECT_ROOT / "datasets"), "none")
    return run_queue(commands, slots, output_dir, show_progress=not args.no_progress)


if __name__ == "__main__":
    raise SystemExit(main())

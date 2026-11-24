# cirkit-circuits-training

Training and evaluation of categorical image circuits with Cirkit, including a
Kaggle notebook for parallel learning-curve experiments on two T4 GPUs.

## Project structure

- `scripts/evaluate_image_circuit.py`: trains and evaluates one experiment.
- `scripts/run_learning_curves.py`: schedules several experiments across GPUs.
- `notebooks/data-driven-initialization.ipynb`: defines the 24-experiment matrix.
- `notebooks/kaggle-parallel-learning-curves.ipynb`: interactive Kaggle entry point.
- `tests/test_parallel_launcher.py`: tests configuration overrides, scheduling and
  progress reporting without requiring GPUs.

## Run on Kaggle

1. Push the current repository changes to GitHub.
2. Upload `notebooks/kaggle-parallel-learning-curves.ipynb` to Kaggle.
3. Enable Internet and select **GPU T4 x2**.
4. Add `WANDB_API_KEY` under **Add-ons > Secrets** and enable it for the notebook.
5. Edit the first configuration cell, then use **Run All**.

The main parameters are:

```python
K_UNITS = 64
TOTAL_EXPERIMENTS = 24
EXPERIMENTS_PER_GPU = 4
MAX_EPOCHS = 100
LEARNING_CURVE_EVERY_STEPS = 100
DASHBOARD_EVERY_STEPS = 10
```

Kaggle supplies two physical T4 GPUs. With `EXPERIMENTS_PER_GPU = 4`, the launcher
runs eight concurrent processes: four share GPU 0 and four share GPU 1. This does
not create eight GPUs. If memory is exhausted or throughput gets worse, reduce the
value to 2 or 1.

The notebook checks the GPU count, installs the repository requirements, authenticates
W&B through the Kaggle secret and previews the exact commands before training.

## Progress and outputs

The launcher displays:

- one overall experiment-completion bar;
- one progress bar for every active GPU slot;
- the experiment assigned to each GPU;
- GPU occupancy, queued experiments and failures.

Every run creates a timestamped directory under `/kaggle/working/results` containing:

- one checkpoint and console log per experiment;
- `run-status.json` with active step counts, GPU assignment and final results.

`--progress-steps` controls only the notebook dashboard. W&B learning curves are
controlled independently by `--validation-steps` and include step 0, periodic
train/validation NLL and BPD, learning rate, and the final step.

## Local usage

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run one data-initialized MNIST experiment:

```bash
python -m scripts.evaluate_image_circuit \
    --dataset mnist --rg quad-tree-2 \
    --input-sharing none --ycc none \
    --activation none --inner-layer cp --use-mixing-weights \
    --use-estimated-weights --alpha 5.0 --noise-scale 2.0 \
    --k 64 --seed 42 --split-seed 42 \
    --max-epochs 100 --no-early-stopping \
    --wandb --validation-steps 100 --progress-steps 10 \
    --save-path mnist_quad-tree-2_data-init_k64_seed42.pt
```

Remove `--use-estimated-weights` for random initialization.

Run the full matrix directly from a two-GPU environment:

```bash
python -m scripts.run_learning_curves \
    --gpus 0 1 \
    --jobs-per-gpu 4 \
    --max-experiments 24 \
    --k 64 \
    --max-epochs 100 \
    --validation-steps 100 \
    --progress-steps 10 \
    --project pc_dataset_eval
```

Use `--dry-run` to inspect all generated commands without downloading data or
starting training.

## Reproducibility

- `--split-seed` fixes the train/validation split.
- `--seed` fixes model initialization and minibatch ordering.
- The notebook can set `REPOSITORY_REF` to a commit SHA.
- Checkpoints, logs and W&B runs are separate for every experiment.

For exact long-term reproduction, pin the Cirkit revision and Python dependency
versions before the final campaign.

## Tests

```bash
python -m unittest tests.test_parallel_launcher -v
```

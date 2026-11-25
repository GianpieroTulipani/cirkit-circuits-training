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
DATASET = 'mnist'
REGION_GRAPHS = ['quad-tree-2', 'quad-tree-4', 'quad-graph', 'chow-liu-tree']
K_UNITS = 64
EXPERIMENTS_PER_GPU = 4
MAX_EPOCHS = 100
LEARNING_CURVE_EVERY_STEPS = 250
DASHBOARD_EVERY_STEPS = 10
WANDB_PROJECT_TEMPLATE = 'pc_{dataset}_{rg}_eval'
```

The dataset is selected before the matrix is scheduled. For each selected region
graph, the launcher runs exactly two experiments (data initialization and random
initialization) and sends them to a dedicated W&B project such as
`pc_mnist_quad-graph_eval`. With the four default RGs, one campaign contains eight
experiments.

Kaggle supplies two physical T4 GPUs. With `EXPERIMENTS_PER_GPU = 4`, the launcher
runs eight concurrent processes: four share GPU 0 and four share GPU 1. This does
not create eight GPUs. If memory is exhausted or throughput gets worse, reduce the
value to 2 or 1.

The notebook checks the GPU count, installs the repository requirements, authenticates
W&B through the Kaggle secret and shows a compact campaign preview before training.

## Progress and outputs

The Kaggle notebook keeps one ordered live dashboard that is replaced every second. It displays:

- GPU 0 and GPU 1 as separate side-by-side sections;
- only the experiments currently assigned to each GPU;
- stable slot numbers and short dataset, structure, initialization, K and seed labels;
- optimizer-step progress, percentage and elapsed time for every active slot;
- compact completed, queued and failed counters.

Training output is redirected to per-experiment files, so notebook cells do not grow
with epoch logs or completed experiments.

Every run creates a timestamped directory under `/kaggle/working/results` containing:

- one checkpoint and console log per experiment;
- `run-status.json` with active step counts, GPU assignment and final results.

`--progress-steps` controls only the notebook dashboard. W&B learning curves are
controlled independently by `--validation-steps` and include step 0 and the final
step. Only two history metrics are sent to W&B: raw validation NLL (`val_nll`) and
validation NLL divided by its value at step 0 (`val_nll_relative`). The raw curve
preserves the initialization advantage; the relative curve starts at 1 for both
runs and makes their optimization speed readable when the raw starting values are
very different.

For presentation, keep the raw panel from step 0 instead of deleting early points.
Use a logarithmic Y axis for the full trajectory, or show it as a small inset next
to a linear zoom of the convergence region. Treat `val_nll_relative` as the second
view: it answers how quickly each optimizer reduces its own starting loss, while
`val_nll` answers which model is actually better at a given step.

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
    --max-epochs 100 --patience 5 \
    --wandb --validation-steps 250 --progress-steps 10 \
    --save-path mnist_quad-tree-2_data-init_k64_seed42.pt
```

Remove `--use-estimated-weights` for random initialization.

Run every initialization pair for one dataset directly from a two-GPU environment:

```bash
python -m scripts.run_learning_curves \
    --gpus 0 1 \
    --jobs-per-gpu 4 \
    --dataset mnist \
    --region-graphs quad-tree-2 quad-tree-4 quad-graph chow-liu-tree \
    --k 64 \
    --max-epochs 100 \
    --validation-steps 250 \
    --patience 5 \
    --progress-steps 10 \
    --project-template 'pc_{dataset}_{rg}_eval'
```

Use `--dry-run` for a compact configuration summary without downloading data or
starting training. Add `--list-commands` only when the full generated command list
is needed. Early stopping remains enabled in every generated command; `--patience`
counts validation checks without improvement, not optimizer steps.

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

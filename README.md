# cirkit-circuits-training

Script for training and evaluating categorical circuits on images using Cirkit's
`data-driven-initialization` branch. The library provides models and initialization;
this project handles data loading, training, and evaluation.

## Installation and usage

From this project's directory:

```bash
python -m pip install -r requirements.txt
python -m scripts.evaluate_image_circuit --help
```

MNIST example using the same arguments as the notebook:

```bash
python -m scripts.evaluate_image_circuit \
    --rg quad-tree-2 --dataset mnist \
    --input-sharing none --ycc none \
    --activation none --inner-layer cp --use-mixing-weights \
    --use-estimated-weights --alpha 5.0 --noise-scale 2.0 \
    --k 128 --seed 42 \
    --save-path mnist_quad-tree-2_data-init_k128_seed42.pt
```

The multiline command uses Bash syntax, also supported in Colab/Kaggle `!` cells.
In PowerShell, run it on a single line.
Remove `--use-estimated-weights` to use random initialization.
For Weights & Biases, also install `wandb` and add `--wandb`.

MNIST, Fashion-MNIST, and CIFAR-10 are downloaded through torchvision.
For local data, use `--dataset tensor --root PATH`: you need `train.pt` and
either `test.pt` or `valid.pt`. If `valid.pt` is missing, the validation set is split
from the training set. If `test.pt` is missing, `valid.pt` is also used for the final
evaluation, as in the original script. Files can contain tensors of shape `(N,C,H,W)`
or `(N,D)`; for the latter, specify `--image-shape C H W`.

## Notebook on Kaggle or Colab

[The updated notebook](notebooks/data-driven-initialization.ipynb) preserves the
experiments and saved outputs from the supplied notebook. The new setup cell
downloads this project, changes to its directory, and installs its dependencies.
The `!python -m scripts.evaluate_image_circuit ...` cells remain unchanged.
The setup cell clones the repository from
`https://github.com/GianpieroTulipani/cirkit-circuits-training.git`: update the URL
if you choose a different name. On Kaggle, enable Internet to clone the repository
and download the data.

There is no need to clone Cirkit separately: `requirements.txt` installs it directly
from the branch. Existing outputs document previous runs; the updated notebook has
not been rerun in full. The setup cell reuses an existing clone without updating it
automatically.

## Linking from the Cirkit branch

The training project depends on the library. There is no need to add the training
project to Cirkit's dependencies or move the script back into the library.
In Cirkit's notebook, you can use the same setup cell as the notebook provided here,
or link directly to this notebook from the documentation:

```markdown
Training and evaluation experiments are available in the
[cirkit-circuits-training project](https://github.com/GianpieroTulipani/cirkit-circuits-training),
with a [notebook for Kaggle and Colab](https://github.com/GianpieroTulipani/cirkit-circuits-training/blob/main/notebooks/data-driven-initialization.ipynb).
```

The notebook already present in the local Cirkit checkout is a different MNIST
example from the one supplied in Downloads. The updated copy here is based on the
supplied file; it does not automatically replace that other example.

"""Train image circuits through the modality API: python -m scripts.evaluate_image_circuit."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import cirkit.symbolic.functional as sf
from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.pipeline import PipelineContext
from cirkit.templates.data_modalities import image_data
from cirkit.templates.initialization import DataDrivenInitialization
from cirkit.templates.utils import POSITIVE_PARAMETER_MIN, Parameterization

logger = logging.getLogger(__name__)


def _log_wandb(values: dict) -> None:
    """Log metrics only when Weights & Biases is explicitly enabled."""
    import wandb

    wandb.log(values)


def _forward_lift(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diff = (y - x).remainder(256)
    average = (x + torch.div(diff, 2, rounding_mode="floor")).remainder(256)
    return average, diff


def rgb_to_ycocg_lossless(images: torch.Tensor) -> torch.Tensor:
    images = images.long()
    red, green, blue = images[:, 0], images[:, 1], images[:, 2]
    tmp, co = _forward_lift(red, blue)
    y, cg = _forward_lift(green, tmp)
    return torch.stack((y, co, cg), dim=1).long()


def rgb_to_ycocg_lossy(images: torch.Tensor) -> torch.Tensor:
    deq_img = (images.float() / 127.5) - 1.0
    red = (deq_img[:, 0] + 1.0) / 2.0
    green = (deq_img[:, 1] + 1.0) / 2.0
    blue = (deq_img[:, 2] + 1.0) / 2.0
    co = red - blue
    tmp = blue + co / 2
    cg = green - tmp
    y = tmp + cg / 2
    y = y * 2.0 - 1.0
    transformed_img = torch.stack((y, co, cg), dim=1)
    return torch.floor(((transformed_img + 1.0) / 2.0) * 256).long().clip(0, 255)


def apply_color_transform(images: torch.Tensor, ycc: str) -> torch.Tensor:
    if ycc == "none":
        return images.long()
    if images.ndim != 4 or images.size(1) != 3:
        shape = tuple(images.shape)
        raise ValueError(
            f"YCoCg transforms require RGB tensors of shape (N, 3, H, W), found {shape}"
        )
    if ycc == "lossy":
        return rgb_to_ycocg_lossy(images)
    if ycc == "lossless":
        return rgb_to_ycocg_lossless(images)
    raise ValueError(f"Unknown YCoCg mode {ycc!r}")


def flatten_images(images: torch.Tensor, ycc: str) -> torch.Tensor:
    images = apply_color_transform(images, ycc)
    return images.reshape(images.size(0), -1).to(torch.uint8)


def _infer_flat_shape(
    data: torch.Tensor, image_shape: tuple[int, int, int] | None
) -> tuple[int, int, int]:
    if image_shape is None:
        raise ValueError("--image-shape C H W is required for flat tensor datasets")
    expected = int(np.prod(image_shape))
    if data.ndim != 2 or data.size(1) != expected:
        raise ValueError(
            f"Expected flat tensor data with shape (N, {expected}) for image_shape={image_shape}, "
            f"found {tuple(data.shape)}"
        )
    return image_shape


def _load_tensor_file(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, TensorDataset):
        tensor = tensor.tensors[0]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, found {type(tensor).__name__}")
    return tensor


def _load_local_tensor_dataset(
    root: str,
    image_shape: tuple[int, int, int] | None,
    ycc: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, tuple[int, int, int]]:
    root_path = Path(root)
    train = _load_tensor_file(root_path / "train.pt")
    valid_path = root_path / "valid.pt"
    test_path = root_path / "test.pt"
    valid = _load_tensor_file(valid_path) if valid_path.exists() else None
    test = _load_tensor_file(test_path) if test_path.exists() else valid
    if test is None:
        raise FileNotFoundError(f"Expected either {valid_path} or {test_path}")

    if train.ndim == 4:
        shape = tuple(train.shape[1:])
        train = flatten_images(train, ycc)
        valid = flatten_images(valid, ycc) if valid is not None and valid.ndim == 4 else valid
        test = flatten_images(test, ycc) if test.ndim == 4 else test
        return train, valid, test, shape

    shape = _infer_flat_shape(train, image_shape)
    if ycc != "none":
        c, h, w = shape
        if c != 3:
            raise ValueError(f"YCoCg transforms require 3 channels, found image_shape={shape}")
        train = flatten_images(train.reshape(-1, c, h, w), ycc)
        valid = flatten_images(valid.reshape(-1, c, h, w), ycc) if valid is not None else None
        test = flatten_images(test.reshape(-1, c, h, w), ycc)
    return train, valid, test, shape


def _load_vision_dataset(
    dataset: str, root: str, ycc: str
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    from torchvision import datasets

    if dataset in {"mnist", "fashion-mnist"}:
        dataset_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_ds = dataset_cls(root=root, train=True, download=True)
        test_ds = dataset_cls(root=root, train=False, download=True)
        return (
            flatten_images(train_ds.data[:, None], ycc),
            flatten_images(test_ds.data[:, None], ycc),
            (1, 28, 28),
        )
    if dataset in {"cifar", "cifar10"}:
        train_ds = datasets.CIFAR10(root=root, train=True, download=True)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True)
        train = torch.as_tensor(train_ds.data).permute(0, 3, 1, 2)
        test = torch.as_tensor(test_ds.data).permute(0, 3, 1, 2)
        return flatten_images(train, ycc), flatten_images(test, ycc), (3, 32, 32)
    raise ValueError(f"Unsupported torchvision dataset {dataset!r}")


def load_discrete_image_dataset(
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    image_shape = tuple(args.image_shape) if args.image_shape is not None else None
    if args.dataset in {"tensor", "imagenet32", "imagenet64", "celeba"}:
        if args.dataset == "imagenet32" and image_shape is None:
            image_shape = (3, 32, 32)
        if args.dataset in {"imagenet64", "celeba"} and image_shape is None:
            image_shape = (3, 64, 64)
        train, valid, test, shape = _load_local_tensor_dataset(args.root, image_shape, args.ycc)
    else:
        train, test, shape = _load_vision_dataset(args.dataset, args.root, args.ycc)
        valid = None

    if valid is None:
        n_val = int(len(train) * args.valid_split)
        n_train = len(train) - n_val
        if n_val <= 0 or n_train <= 0:
            raise ValueError(
                f"valid_split={args.valid_split} gives train/val sizes {n_train}/{n_val}; "
                "use a larger dataset or a split in (0, 1)"
            )
        indices = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(args.split_seed)
        )
        train, valid = train[indices[:n_train]], train[indices[n_train:]]

    return train.to(torch.uint8), valid.to(torch.uint8), test.to(torch.uint8), shape


@torch.inference_mode()
def _mean_nll(nll_fn, data_loader, device: torch.device) -> float:
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    sample_count = 0
    for (raw_batch,) in data_loader:
        batch = raw_batch.to(device).long()
        loss_sum.add_(nll_fn(batch).to(torch.float64), alpha=batch.size(0))
        sample_count += batch.size(0)
    return loss_sum.item() / sample_count


def _sum_parameters_to_project(circuit, activation: str) -> list[torch.nn.Parameter]:
    """Return each raw sum parameter once; input logits must remain unconstrained."""
    if activation != "none":
        return []
    parameters = {}
    for layer in circuit.layers:
        if not isinstance(layer, TorchInputLayer):
            for parameter in layer.parameters():
                parameters[id(parameter)] = parameter
    return list(parameters.values())


@torch.no_grad()
def _project_positive(parameters: list[torch.nn.Parameter]) -> None:
    """Project raw positive parameters away from zero before logarithms are taken."""
    for parameter in parameters:
        parameter.clamp_(min=POSITIVE_PARAMETER_MIN)


def train_circuit(
    circuit, nll_fn, train_loader, val_loader, device, args, num_dimensions
):
    """Train with the CLI settings, saving the best validation checkpoint."""
    max_steps = len(train_loader) * args.max_epochs
    if args.max_steps is not None:
        max_steps = min(max_steps, args.max_steps)
    # Project only raw sum weights; categorical input logits stay unconstrained.
    projected_sum_params = _sum_parameters_to_project(circuit, args.activation)
    _project_positive(projected_sum_params)
    optimizer = optim.Adam(
        circuit.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=1, eta_min=args.eta_min
        )

    best_val_nll = float("inf")
    checks_no_improve = 0
    total_steps = 0
    avg_val_nll = None
    saved_best = False

    def emit_progress(step: int) -> None:
        """Emit a stable line that the multi-GPU launcher can parse."""
        print(
            f"EXPERIMENT_PROGRESS step={step} max_steps={max_steps}",
            flush=True,
        )

    # Evaluate the same small training sample at every point, at the current weights.
    # A separate generator keeps evaluation from consuming the training RNG stream.
    if args.wandb:
        train_data = train_loader.dataset.tensors[0]
        probe_indices = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(args.split_seed)
        )[:1024]
        train_eval_loader = DataLoader(
            TensorDataset(train_data[probe_indices]),
            batch_size=args.batch_size,
            generator=torch.Generator().manual_seed(args.split_seed),
        )

    def log_curves(step: int, val_nll: float) -> None:
        train_nll = _mean_nll(nll_fn, train_eval_loader, device)
        _log_wandb(
            {
                "step": step,
                "train_nll": train_nll,
                "train_bpd": train_nll / (num_dimensions * np.log(2.0)),
                "val_nll": val_nll,
                "val_bpd": val_nll / (num_dimensions * np.log(2.0)),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

    if args.wandb:
        circuit.eval()
        log_curves(0, _mean_nll(nll_fn, val_loader, device))

    print(f"Starting training for a maximum of {max_steps} steps.\n")
    emit_progress(0)

    while total_steps < max_steps and (
        args.no_early_stopping or checks_no_improve < args.patience
    ):
        circuit.train()
        # Keep detached metrics on-device; .item() per batch would synchronize CUDA.
        train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        train_count = 0
        for (raw_batch,) in tqdm(train_loader, desc="[Train]", leave=False):
            batch = raw_batch.to(device).long()

            loss = nll_fn(batch)

            train_loss_sum.add_(loss.detach().to(torch.float64), alpha=batch.size(0))
            train_count += batch.size(0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            _project_positive(projected_sum_params)

            total_steps += 1

            if total_steps % args.progress_steps == 0 or total_steps == max_steps:
                emit_progress(total_steps)

            if total_steps % args.validation_steps == 0 or total_steps == max_steps:
                circuit.eval()
                avg_train_nll = train_loss_sum.item() / train_count
                avg_val_nll = _mean_nll(nll_fn, val_loader, device)
                bpd_train = avg_train_nll / (num_dimensions * np.log(2.0))
                bpd_val = avg_val_nll / (num_dimensions * np.log(2.0))

                if args.wandb:
                    log_curves(total_steps, avg_val_nll)

                # Keep the original tolerance and tie handling for checkpoint selection.
                if avg_val_nll - args.min_delta <= best_val_nll:
                    best_val_nll = avg_val_nll
                    torch.save(circuit.state_dict(), args.save_path)
                    saved_best = True
                    checks_no_improve = 0
                    logger.info(
                        "New best model at step %d, Train NLL: %.4f, Train bpd: %.4f, "
                        "Val NLL: %.4f, Val bpd: %.4f",
                        total_steps,
                        avg_train_nll,
                        bpd_train,
                        best_val_nll,
                        bpd_val,
                    )
                else:
                    checks_no_improve += 1
                    logger.info(
                        "No improvement at step %d, count: %d/%d",
                        total_steps,
                        checks_no_improve,
                        args.patience,
                    )

                if not args.no_early_stopping and checks_no_improve >= args.patience:
                    logger.info("Early stopping triggered.")
                    break

                circuit.train()

            if total_steps >= max_steps:
                break

    if not saved_best:
        torch.save(circuit.state_dict(), args.save_path)
        logger.info(
            "No validation checkpoint was saved; saved current model to %s.", args.save_path
        )

    if args.wandb:
        metrics = {"saved_best_checkpoint": saved_best}
        if avg_val_nll is not None:
            metrics.update(
                final_online_train_nll=avg_train_nll,
                final_online_train_bpd=bpd_train,
                final_val_nll=avg_val_nll,
                final_val_bpd=bpd_val,
                best_val_nll=best_val_nll,
            )
        _log_wandb(metrics)


@torch.inference_mode()
def evaluate_circuit(
    circuit, nll_fn, test_loader, device, num_dimensions, log_to_wandb=False
):
    """Report test NLL and bits per dimension using the loaded checkpoint."""
    circuit.eval()

    avg_test_nll = _mean_nll(nll_fn, tqdm(test_loader, desc="[Test]", leave=False), device)
    avg_test_bpd = avg_test_nll / (num_dimensions * np.log(2.0))

    logger.info("Test NLL: %.4f | bpd: %.4f", avg_test_nll, avg_test_bpd)

    if log_to_wandb:
        _log_wandb({"test_nll": avg_test_nll, "test_bpd": avg_test_bpd})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PC on a discrete image dataset")
    add = parser.add_argument

    # Data and model
    add(
        "--dataset",
        default="mnist",
        choices=[
            "mnist",
            "fashion-mnist",
            "cifar",
            "cifar10",
            "celeba",
            "imagenet32",
            "imagenet64",
            "tensor",
        ],
    )
    add("--root", default="datasets")
    add("--image-shape", type=int, nargs=3, metavar=("C", "H", "W"))
    add("--ycc", default="none", choices=["none", "lossy", "lossless"])
    add(
        "--input-sharing",
        default="none",
        choices=["none", "full"],
        help="Full sharing on multichannel images requires a quad-tree or quad-graph RG.",
    )
    add(
        "--rg",
        default="quad-tree-2",
        choices=[
            "quad-tree-2",
            "quad-tree-4",
            "quad-graph",
            "random-binary-tree",
            "poon-domingos",
            "chow-liu-tree",
            "linear-tree",
        ],
        help="Region graph; chow-liu-tree learns its structure from the training split.",
    )
    add(
        "--num-bins",
        type=int,
        default=None,
        help=(
            "Chow-Liu structure learning only: bin pixel values into 1..256 categories. "
            "Defaults to all 256 levels; input distributions still use 256 categories."
        ),
    )
    add(
        "--mi-chunk-size",
        type=int,
        default=None,
        help=(
            "Chow-Liu only: positive sample count per mutual-information chunk. "
            "Defaults to all training samples; does not reduce the joint-count table size."
        ),
    )
    add("--inner-layer", default="cp", choices=["cp", "cp-t", "tucker"])
    add("--k", type=int, default=64)
    add(
        "--activation",
        default="none",
        choices=["none", "clamp", "softmax", "softplus", "sigmoid"],
        help=(
            "Sum-weight activation; categorical inputs always use softmax. 'none' projects "
            "only raw sum weights to positive values before training and after optimizer "
            "steps; 'clamp' uses Cirkit's local positive-clamp parameter activation."
        ),
    )
    add(
        "--weights-init",
        default="uniform",
        choices=["uniform", "normal", "dirichlet"],
        help="Sum-weight initialization; input logits use normal initialization unless estimated.",
    )

    # Training
    add("--lr", type=float, default=0.01)
    add("--T_0", type=int, default=1)
    add("--eta-min", type=float, default=0.0001)
    add("--weight-decay", type=float, default=0.0)
    add("--batch-size", type=int, default=256)
    add("--max-epochs", type=int, default=100)
    add("--max-steps", type=int, help="Optional cap on optimizer updates.")
    add(
        "--validation-steps", type=int, default=100,
        help="Evaluate and log learning curves every N optimizer updates.",
    )
    add(
        "--progress-steps", type=int, default=10,
        help="Emit launcher progress every N optimizer updates without logging W&B metrics.",
    )
    add("--no-early-stopping", action="store_true", help="Train for the full step/epoch budget.")
    add("--patience", type=int, default=5)
    add("--min-delta", type=float, default=0.0)
    add("--valid-split", type=float, default=0.05)
    add("--use-scheduler", action="store_true")

    # Initialization
    add("--alpha", type=float, default=5.0)
    add("--noise-scale", type=float, default=2.0)
    add("--use-mixing-weights", action="store_true")
    add("--use-estimated-weights", action="store_true")

    # Optional features
    add(
        "--torch-compile",
        action="store_true",
        help="Use Cirkit's torch-compile backend instead of the standard torch backend.",
    )
    add("--wandb", action="store_true")
    add("--project", default="pc_dataset_eval")

    add("--seed", type=int, default=42)
    add("--split-seed", type=int, help="Fixed train/validation split seed; defaults to --seed.")
    add("--save-path", default="best_circuit.pt")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.validation_steps <= 0 or args.progress_steps <= 0 or args.max_epochs <= 0:
        parser.error("--validation-steps, --progress-steps and --max-epochs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.patience <= 0 and not args.no_early_stopping:
        parser.error("--patience must be positive when early stopping is enabled")
    if args.num_bins is not None and not 1 <= args.num_bins <= 256:
        parser.error("--num-bins must be between 1 and 256")
    if args.mi_chunk_size is not None and args.mi_chunk_size <= 0:
        parser.error("--mi-chunk-size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    print(f"  device: {device}")
    print("=" * 70)

    if args.wandb:
        import wandb

        method = "data-init" if args.use_estimated_weights else "random-init"
        comparison = f"{args.dataset}_{args.rg}_k{args.k}"
        run = wandb.init(
            project=args.project,
            name=f"{comparison}_{method}_seed{args.seed}",
            group=comparison,
            job_type=method,
            config={**vars(args), "initialization": method, "train_curve_max_samples": 1024},
        )
        run.define_metric("step")
        for metric in ("train_nll", "train_bpd", "val_nll", "val_bpd", "learning_rate"):
            run.define_metric(metric, step_metric="step")
    else:
        run = None

    train_data, val_data, test_data, image_shape = load_discrete_image_dataset(args)
    if (
        image_shape[0] > 1
        and args.input_sharing == "full"
        and args.rg not in {"quad-tree-2", "quad-tree-4", "quad-graph"}
    ):
        parser.error(
            "Multichannel --input-sharing full requires a quad-tree or quad-graph RG; "
            "use --input-sharing none with this RG"
        )
    logger.info(
        "Loaded %s: train=%s, val=%s, test=%s",
        args.dataset,
        tuple(train_data.shape),
        tuple(val_data.shape),
        tuple(test_data.shape),
    )
    logger.info(
        "Using image_shape=%s, ycc=%s, input_sharing=%s",
        image_shape,
        args.ycc,
        args.input_sharing,
    )
    train_loader = DataLoader(
        TensorDataset(train_data), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        TensorDataset(val_data), batch_size=args.batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(args.split_seed),
    )
    test_loader = DataLoader(
        TensorDataset(test_data), batch_size=args.batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    logger.info("Using %d samples to build/estimate the circuit", len(train_data))
    initialization = (
        DataDrivenInitialization(
            alpha=args.alpha,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )
        if args.use_estimated_weights
        else None
    )
    parameter_activation = "positive-clamp" if args.activation == "clamp" else args.activation
    sum_weight_param = Parameterization(
        activation=parameter_activation,
        initialization=args.weights_init,
    )
    symbolic_circuit = image_data(
        image_shape,
        data=train_data.to(device),
        input_layer="categorical",
        region_graph=args.rg,
        num_bins=args.num_bins,
        mi_chunk_size=args.mi_chunk_size,
        sum_weight_param=sum_weight_param,
        sum_product_layer=args.inner_layer,
        num_input_units=args.k,
        num_sum_units=args.k,
        use_mixing_weights=args.use_mixing_weights,
        initialization=initialization,
        input_sharing=args.input_sharing,
    )

    symbolic_partition_function = sf.integrate(symbolic_circuit)

    logger.info("Circuit built with %d layers", len(list(symbolic_circuit.layers)))

    backend = "torch-compile" if args.torch_compile else "torch"
    logger.info("Using Cirkit backend=%s", backend)
    ctx = PipelineContext(backend=backend, semiring="lse-sum", fold=True, optimize=True)

    circuit = ctx.compile(symbolic_circuit).to(device)
    circuit_partition_function = ctx.compile(symbolic_partition_function).to(device)

    def nll(batch):
        # Recompute log Z with the current parameters, also after loading a checkpoint.
        return -(circuit(batch) - circuit_partition_function()).mean()

    logger.info("Number of parameters: %d", sum(p.numel() for p in circuit.parameters()))

    num_dimensions = int(np.prod(image_shape))
    train_circuit(
        circuit, nll, train_loader, val_loader, device, args, num_dimensions
    )

    circuit.load_state_dict(torch.load(args.save_path, map_location=device, weights_only=True))

    evaluate_circuit(
        circuit, nll, test_loader, device, num_dimensions, log_to_wandb=args.wandb
    )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

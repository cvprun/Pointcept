# -*- coding: utf-8 -*-

"""Pointcept PTv3 semantic-segmentation trainer — training-wizard entrypoint.

Serves the ``pointcept`` framework for ``semantic_segmentation``. Like
``yolo_detect.py`` and ``rfdetr_detect.py`` it is held to the three rules any
partner image is held to (``docs/training_trainer_contract.md``): log metrics
to MLflow, call ``pyfunc.log_model()`` once, exit 0 or non-zero.

So it does **not**: download the dataset (the platform stages it into
``GEO_DATA_DIR``), emit step events, announce its MLflow run, register the
model, or exit with a particular code on SIGTERM.

This is the first trainer whose input is a point cloud, and three consequences
follow from that:

* Labels live *inside* the file, as an integer vertex property, so there are no
  annotation files to parse. The property is chosen by the same rule the
  platform's validator used (:func:`label_property`) -- picking a different
  column than the one the wizard listed classes from would train on something
  nobody chose. CloudCompare stores that property as ``float scalar_<name>``;
  the same helper the validator uses turns it back into integers.
* Pointcept reads a directory of ``.npy`` arrays, not ``.ply``, so the staged
  files are **converted**, not symlinked as the image trainers do.
* Pointcept is configured by a whole python file rather than keyword arguments.
  The trainer writes one into ``GEO_WORK_DIR`` and it ships as a run artifact,
  so what actually ran is recoverable.

Four things about Pointcept are load-bearing here and were each verified
against the real package (plan 36 Phase 0):

* ``Trainer.train()`` runs inside ``ExceptionWriter``, which turns **any**
  exception into ``sys.exit(1)`` after printing a traceback. Raising to break
  out of the epoch loop would therefore report a crash and skip
  ``after_train``, so stopping is a ``Trainer`` subclass that breaks between
  epochs instead. Shrinking ``max_epoch`` does not work either: the loop's
  ``range()`` is evaluated once.
* Pointcept already computes every metric worth having and hands it to
  ``trainer.writer``. The hook swaps that writer for a proxy rather than
  recomputing anything, so the platform's numbers and Pointcept's own log can
  never disagree. A ``None`` writer means the code that feeds it is skipped
  outright, so installing the proxy *is* the enable switch -- tensorboard stays
  off.
* ``CheckpointLoader`` loads with ``strict=False``, and ``strict=False``
  forgives a missing key but **not** a shape mismatch: a 20-class ScanNet head
  stops a 3-class run dead. Starting weights are therefore filtered to the
  tensors that fit before Pointcept sees them.
* That same loader treats a missing weight file as "no weights" and trains from
  scratch, reporting success. A checkpoint the submitter chose is never allowed
  to vanish that quietly.

Runs as the Pointcept runtime image's ``ENTRYPOINT`` (``Dockerfile.pointcept``
next to this file).
"""

import argparse
import importlib.util
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import annotations

#: Scratch space and the read-only staged dataset (trainer contract §1.1).
WORKDIR = annotations.WORKDIR
DATA_DIR = annotations.DATA_DIR

#: Where the ``DefaultDataset`` tree we build for Pointcept lands.
DATASET_DIR = WORKDIR / "data" / "pointcept"

#: Where Pointcept writes ``model/*.pth``, ``train.log`` and ``config.py``.
OUTPUT_DIR = WORKDIR / "runs" / "pointcept"

#: Where the platform's stager leaves a pretrained checkpoint
#: (``training/weights.py::DEST_RULES``) and where a filtered copy is written.
WEIGHTS_DIR = WORKDIR / "weights"

#: Pointcept's "no label here" marker. Points whose label is not one of the
#: selected classes get this rather than a class of their own, so the model is
#: never taught a category nobody asked for.
IGNORE_INDEX = -1

#: Substrings marking a PLY vertex property as a semantic label, copied from
#: ``services/dataset_files.py``. **This has to match**: the wizard offers the
#: values that file's validator found, and a trainer reading a different
#: property would train on data the submitter never saw.
LABEL_NAME_TOKENS = ("label", "class", "segment", "semantic")

#: CloudCompare's prefix on every exported scalar field (same source).
SCALAR_PREFIX = "scalar_"

#: Properties that are geometry or appearance and are never labels (same
#: source). Used only for the fallback scan, when no property carries a known
#: label name.
GEOMETRY_PROPERTY_NAMES = {
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
    "alpha",
}

#: Catalog model key -> the public checkpoint it starts from. A mirror of the
#: ``weights`` asset in ``training/catalog_data.py``, kept here for the same
#: reason ``rfdetr_detect.MODEL_CLASSES`` mirrors the catalog: the trainer runs
#: in a container that has no access to the server's tables.
#:
#: Unlike rfdetr and ultralytics, Pointcept has no idea which checkpoint is
#: "its own" -- there is no resolver to hand a filename to. So the trainer both
#: looks where the stager puts the file and, on a mirror miss, fetches it
#: itself (plan 37 decision 7: a mirror miss is slower, not broken).
MODEL_WEIGHTS = {
    "pt-v3": {
        "filename": "ptv3-scannet-base.pth",
        "url": (
            "https://huggingface.co/Pointcept/PointTransformerV3/resolve/main"
            "/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"
        ),
        "md5": "c8f558b7f6a159f9d60c11efad84fe41",
    },
}

#: Pointcept's own scalar names -> the platform's. The charts, the "primary
#: metric" ranking and the progress card all key off ``metrics/*``
#: (``training/metrics_catalog.py``), so these four are written twice: once
#: under Pointcept's name for anyone comparing with its log, once under ours.
METRIC_ALIASES = {
    "val/mIoU": "metrics/mIoU",
    "val/mAcc": "metrics/mAcc",
    "val/allAcc": "metrics/allAcc",
    "val/macroF1": "metrics/macroF1",
}

#: System packages a serving image would need on top of MLflow's python-slim
#: base. open3d dlopens libGL at import and Pointcept's dataset package imports
#: it unconditionally.
SERVING_APT_PACKAGES = ["libgl1", "libglib2.0-0", "libgomp1"]

#: Fraction of scenes held out when the submission asks for a random split and
#: gives no percentage. Matches the wizard's own default.
DEFAULT_TRAIN_PERCENT = 80


class StopTraining(Exception):
    """Raised at a safe point once SIGTERM has been seen."""


#: Set by the SIGTERM handler; polled at epoch boundaries. Handling it as a flag
#: rather than dying in the handler is what makes the checkpoint survive.
stop_requested = False


def _on_sigterm(_signum, _frame) -> None:
    global stop_requested
    stop_requested = True
    print("-> SIGTERM received; will stop at the next epoch boundary", flush=True)


def install_sigterm_flag() -> None:
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # pragma: no cover - not on the main thread
        pass


# --- staged PLY -> Pointcept's DefaultDataset layout -------------------------


def is_label_name(name: str) -> bool:
    """Validator's rule: a label-ish word in the name, ``scalar_`` stripped."""

    lowered = name.lower()
    if lowered.startswith(SCALAR_PREFIX):
        lowered = lowered[len(SCALAR_PREFIX) :]
    return any(token in lowered for token in LABEL_NAME_TOKENS)


def label_property(names, vertex=None) -> "str | None":
    """The vertex property holding semantic labels, by the validator's rule.

    A label-named property wins; otherwise (when ``vertex`` is given) the first
    integer property that is not geometry. Returns ``None`` when the file
    carries no labels at all -- which is not an error here, only upstream (a
    dataset of unlabelled clouds offers no classes, and submission refuses it
    before this code runs).
    """

    for name in names:
        if is_label_name(name):
            return name
    if vertex is None:
        return None
    import numpy as np

    for name in names:
        if name.lower() in GEOMETRY_PROPERTY_NAMES:
            continue
        if np.issubdtype(vertex.dtype[name], np.integer):
            return name
    return None


def integer_labels(values):
    """Validator's rule: ``int64`` for integer columns and for float columns
    of whole numbers (CloudCompare), ``None`` for anything else."""

    import numpy as np

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and np.all(np.isfinite(arr)) and np.all(arr == np.rint(arr)):
            return np.rint(arr).astype(np.int64)
    return None


def read_ply(path: Path):
    """``(coord, color, label)`` for one file; ``color``/``label`` may be None."""

    import numpy as np
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = list(vertex.dtype.names or ())
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"{path.name}: PLY vertex has no '{axis}' property")

    coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    color = None
    if {"red", "green", "blue"}.issubset(names):
        color = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=1
        ).astype(np.float32)
    name = label_property(names, vertex)
    label = None
    if name is not None:
        label = vertex[name]
        as_int = integer_labels(label)
        if as_int is not None:
            label = as_int
    return coord, color, label


def split_scenes(paths: list, split: dict) -> tuple:
    """Hold scenes out for validation — a point cloud is one sample.

    Splitting inside a scene would put neighbouring points on both sides and
    report a validation score the model never earned.
    """

    import numpy as np

    percent = int(split.get("train_percent") or DEFAULT_TRAIN_PERCENT)
    seed = int(split.get("seed") or 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    # At least one scene on each side: Pointcept builds a validation loader
    # unconditionally and an empty one fails at the first evaluation.
    n_train = min(max(1, round(len(paths) * percent / 100)), max(1, len(paths) - 1))
    train_idx = set(order[:n_train].tolist())
    train = [paths[i] for i in sorted(train_idx)]
    val = [paths[i] for i in range(len(paths)) if i not in train_idx]
    return train, val


def build_dataset(
    paths: list, classes: list, split: dict, root: "Path | None" = None
) -> tuple:
    """Write ``{train,val}/{scene}/{coord,color,segment}.npy``.

    ``classes`` are the label values the submitter chose, in the order they
    chose them; that order **is** the class index, and the same list is carried
    into the model's metadata so serving can name what it predicts.

    Returns ``(root, has_color)`` -- a dataset without colour trains on
    coordinates alone rather than on a fabricated constant.
    """

    import numpy as np

    root = root or DATASET_DIR
    remap = {}
    for index, value in enumerate(classes):
        try:
            remap[int(value)] = index
        except (TypeError, ValueError):
            raise ValueError(
                f"class '{value}' is not a point-cloud label value; the wizard "
                "lists them as integers"
            )

    train, val = split_scenes(paths, split)
    has_color = True
    counts = {"train": 0, "val": 0}
    for part, members in (("train", train), ("val", val)):
        for path in members:
            if stop_requested:
                raise StopTraining("stopped during data preparation")
            coord, color, label = read_ply(path)
            scene = root / part / path.stem
            scene.mkdir(parents=True, exist_ok=True)
            np.save(scene / "coord.npy", coord)
            if color is None:
                has_color = False
            else:
                np.save(scene / "color.npy", color)
            segment = np.full(len(coord), IGNORE_INDEX, dtype=np.int32)
            if label is not None:
                for value, index in remap.items():
                    segment[np.asarray(label) == value] = index
            np.save(scene / "segment.npy", segment)
            counts[part] += 1

    print(
        f"-> prepared {counts['train']} train / {counts['val']} val scene(s)"
        f"{'' if has_color else ' (no colour; training on coordinates alone)'}",
        flush=True,
    )
    return root, has_color


# --- configuration -----------------------------------------------------------


def _hyperparameters(config: dict) -> dict:
    """The submission's hyperparameters, with this framework's defaults."""

    values = dict(config.get("hyperparameters") or {})
    defaults = {
        "epochs": 100,
        "batch_size": 2,
        "num_workers": 4,
        "lr": 0.006,
        "weight_decay": 0.05,
        "grid_size": 0.6,
        "mix_prob": 0.0,
        "drop_path": 0.3,
        "patch_size": 1024,
        "enable_amp": True,
        "spconv_native": False,
    }
    return {**defaults, **{k: v for k, v in values.items() if v is not None}}


def _flash_attention_available() -> bool:
    """Is ``flash_attn`` in this image?

    PTv3 runs without it, but then it materialises the whole attention matrix
    and a 24k-point scene at batch 2 exhausts an 8 GB card (plan 36 Phase 0), so
    the runtime image is built to carry it.

    Some targets cannot carry it. flash-attn 2.x ships cubins for sm_80/90/100
    /120 only, and Blackwell splits into families: an sm_120 cubin does not load
    on sm_121 (DGX Spark). The image builder asks nvcc for the family's PTX so
    the driver can JIT it, but that is a best effort -- when it does not land,
    ``enable_flash=True`` stops training at ``assert flash_attn is not None``
    while it is building the model. Running slowly beats not running.
    """

    return importlib.util.find_spec("flash_attn") is not None


def build_config_text(
    data_root: Path,
    save_path: Path,
    class_names: list,
    params: dict,
    *,
    has_color: bool,
    weight: str = "",
) -> str:
    """The Pointcept config for this run, as the text of a config file.

    Written flat rather than inheriting ``_base_``: the file is logged as a run
    artifact, and a config that only makes sense next to the repo it came from
    would not explain a finished run six months later.
    """

    feat_keys = ("coord", "color") if has_color else ("coord",)
    num_classes = len(class_names)

    # Recorded in the config text rather than left implicit: this is the single
    # switch that decides whether a run was fast or merely finished.
    enable_flash = _flash_attention_available()
    if not enable_flash:
        print(
            "-> flash-attn is not in this image; PTv3 falls back to dense "
            "attention (slower, and far heavier on memory)",
            flush=True,
        )
    grid_size = float(params["grid_size"])
    patch_size = int(params["patch_size"])
    lr = float(params["lr"])
    epochs = int(params["epochs"])

    color_aug = (
        [
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
        ]
        if has_color
        else []
    )
    normalize = [dict(type="NormalizeColor")] if has_color else []

    train_transform = [
        dict(type="CenterShift", apply_z=True),
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        *color_aug,
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="CenterShift", apply_z=False),
        *normalize,
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment"),
            feat_keys=feat_keys,
        ),
    ]
    val_transform = [
        dict(type="CenterShift", apply_z=True),
        dict(type="Copy", keys_dict={"segment": "origin_segment"}),
        dict(
            type="GridSample",
            grid_size=grid_size,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
            return_inverse=True,
        ),
        dict(type="CenterShift", apply_z=False),
        *normalize,
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
            feat_keys=feat_keys,
        ),
    ]

    cfg = dict(
        weight=weight or None,
        resume=False,
        evaluate=True,
        test_only=False,
        seed=0,
        save_path=str(save_path),
        num_worker=int(params["num_workers"]),
        batch_size=int(params["batch_size"]),
        batch_size_val=1,
        batch_size_test=1,
        epoch=epochs,
        # Pointcept's data loop is ``epoch // eval_epoch``; keeping them equal
        # means one pass over the scenes per epoch, which is what the submitted
        # "epochs" is understood to mean everywhere else in the platform.
        eval_epoch=epochs,
        gradient_accumulation_steps=1,
        clip_grad=None,
        sync_bn=False,
        enable_amp=bool(params["enable_amp"]),
        amp_dtype="float16",
        # Not a Pointcept field -- ``GeoSpconvAlgo`` reads it off the config,
        # and recording it here keeps the artifact a full account of the run.
        spconv_native=bool(params["spconv_native"]),
        empty_cache=False,
        empty_cache_per_epoch=False,
        find_unused_parameters=False,
        # The platform owns the run; Pointcept's own logger would open a second
        # one somewhere nobody looks.
        enable_wandb=False,
        mix_prob=float(params["mix_prob"]),
        param_dicts=[dict(keyword="block", lr=lr / 10.0)],
        model=dict(
            type="DefaultSegmentorV2",
            num_classes=num_classes,
            backbone_out_channels=64,
            backbone=dict(
                type="PT-v3m1",
                in_channels=3 * len(feat_keys),
                order=("z", "z-trans", "hilbert", "hilbert-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(patch_size,) * 5,
                dec_depths=(2, 2, 2, 2),
                dec_channels=(64, 64, 128, 256),
                dec_num_head=(4, 4, 8, 16),
                dec_patch_size=(patch_size,) * 4,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=float(params["drop_path"]),
                shuffle_orders=True,
                pre_norm=True,
                enable_rpe=False,
                # Wanted, not assumed: without flash attention PTv3
                # materialises the whole attention matrix and a 24k-point scene
                # at batch 2 exhausts an 8 GB card (plan 36 Phase 0). Not every
                # GPU family has a flash-attn build though --
                # :func:`_flash_attention_available` explains which.
                enable_flash=enable_flash,
                upcast_attention=False,
                upcast_softmax=False,
                enc_mode=False,
                pdnorm_bn=False,
                pdnorm_ln=False,
            ),
            criteria=[
                dict(
                    type="CrossEntropyLoss",
                    loss_weight=1.0,
                    ignore_index=IGNORE_INDEX,
                ),
                dict(
                    type="LovaszLoss",
                    mode="multiclass",
                    loss_weight=1.0,
                    ignore_index=IGNORE_INDEX,
                ),
            ],
        ),
        optimizer=dict(type="AdamW", lr=lr, weight_decay=float(params["weight_decay"])),
        scheduler=dict(
            type="OneCycleLR",
            max_lr=[lr, lr / 10.0],
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        ),
        data=dict(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            names=list(class_names),
            train=dict(
                type="DefaultDataset",
                split="train",
                data_root=str(data_root),
                transform=train_transform,
                test_mode=False,
            ),
            val=dict(
                type="DefaultDataset",
                split="val",
                data_root=str(data_root),
                transform=val_transform,
                test_mode=False,
            ),
        ),
        hooks=[
            dict(type="CheckpointLoader"),
            dict(type="IterationTimer", warmup_iter=2),
            dict(type="InformationWriter"),
            dict(type="GeoSpconvAlgo"),
            dict(type="SemSegEvaluator"),
            dict(type="CheckpointSaver", save_freq=None),
            dict(type="GeoProgress"),
        ],
        train=dict(type="GeoTrainer"),
    )

    lines = [
        "# Generated by the platform's Pointcept trainer (plan 36).",
        "# Edited copies are not read back -- this file is a record of the run.",
        "",
    ]
    for key, value in cfg.items():
        lines.append(f"{key} = {value!r}")
    return "\n".join(lines) + "\n"


# --- MLflow ------------------------------------------------------------------


def _run_id() -> "str | None":
    from_env = os.environ.get("MLFLOW_RUN_ID")
    if from_env:
        return from_env
    try:
        import mlflow

        run = mlflow.active_run() or mlflow.last_active_run()
        return run.info.run_id if run is not None else None
    except Exception:  # noqa: BLE001 - tracing must never kill training
        return None


def _client():
    """An ``MlflowClient``, or ``None`` when there is nothing to write to."""

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        from mlflow import MlflowClient

        return MlflowClient()
    except Exception as exc:  # noqa: BLE001 - tracing must never kill training
        print(f"-> mlflow client unavailable: {exc}", flush=True)
        return None


def _log_run_artifact(local_path, artifact_path: str = "pointcept") -> "str | None":
    """Attach a file to the run. Never raises."""

    run_id = _run_id()
    if not run_id or not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow

        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        return f"{artifact_path}/{Path(local_path).name}"
    except Exception as exc:  # noqa: BLE001 - upload must not fail the run
        print(f"-> artifact upload failed: {exc}", flush=True)
        return None


# --- hook + trainer ----------------------------------------------------------


def register_extensions(client, run_id):
    """Register the metric hook and the stoppable trainer with Pointcept.

    Both go in through the registries Pointcept builds the run from
    (``HOOKS`` / ``TRAINERS``), named from the generated config -- nothing
    private is patched.
    """

    from pointcept.engines.hooks import HOOKS, HookBase
    from pointcept.engines.train import TRAINERS, Trainer
    from pointcept.utils import comm
    from pointcept.utils.events import EventStorage, ExceptionWriter

    class _MlflowWriter:
        """Stands in for the tensorboard writer and forwards to MLflow.

        Pointcept already computes ``val/mIoU``, ``val/mAcc``, ``val/allAcc``,
        ``val/macroF1``, ``val/loss`` and ``train/loss`` and hands each to
        ``trainer.writer``; taking them here means the platform never
        recomputes a metric and can never disagree with Pointcept's own log.

        Only epoch-stepped series are forwarded. ``train_batch/*`` and
        ``params/lr`` arrive once per iteration, and mixing two step axes in
        one run makes the charts and the progress bar meaningless.
        """

        def __init__(self):
            self.written = 0

        def add_scalar(self, tag, value, step=None, *args, **kwargs):
            tag = str(tag)
            if not tag.startswith(("train/", "val/")):
                return
            if client is None or not run_id:
                return
            # Pointcept counts epochs from 1; the platform's contract is that a
            # metric's step *is* the epoch, 0-based, and its progress bar reads
            # ``max(step) + 1``. One off here shows a run as a step ahead of
            # itself for its whole life.
            index = max(int(step or 1) - 1, 0)
            for name in (tag, METRIC_ALIASES.get(tag)):
                if name is None:
                    continue
                try:
                    client.log_metric(run_id, name, float(value), step=index)
                    self.written += 1
                except Exception as exc:  # noqa: BLE001 - never kill training
                    print(f"-> metric {name} not logged: {exc}", flush=True)

        def add_histogram(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    @HOOKS.register_module()
    class GeoSpconvAlgo(HookBase):
        """Pin every sparse convolution to spconv's ``Native`` algorithm.

        spconv picks ``MaskImplicitGemm`` for every kernel this model uses
        (``spconv/pytorch/conv.py``), and those kernels are the ones that die
        on a node holding no cubin for its own GPU. cumm accepts no
        architecture past 12.0, so an sm_121 target is built as ``12.0+PTX``
        and the driver JIT compiles it (``scripts/build_wheels.sh``); the
        result faults at a different iteration every run, in training or in
        evaluation, at any scene size -- the signature of a bad kernel rather
        than of bad data.

        Switching after the model is built is safe *because* spconv stores
        every weight as KRSC regardless of algorithm (``ALL_WEIGHT_IS_KRSC``
        is True in this build); the layout branch at ``conv.py:126`` is dead
        code here. If that ever stops holding, this hook silently trains a
        model whose weights are laid out for the wrong kernel, so it asserts
        rather than trusting it.
        """

        def before_train(self):
            if not bool(getattr(self.trainer.cfg, "spconv_native", False)):
                return

            from spconv.constants import ALL_WEIGHT_IS_KRSC
            from spconv.core import ConvAlgo
            from spconv.pytorch.conv import SparseConvolution

            assert ALL_WEIGHT_IS_KRSC, (
                "this spconv build lays weights out per algorithm; pinning the "
                "algorithm after the model was built would mismatch them"
            )
            pinned = 0
            for module in self.trainer.model.modules():
                if isinstance(module, SparseConvolution):
                    module.algo = ConvAlgo.Native
                    pinned += 1
            print(
                f"-> spconv: {pinned} layer(s) pinned to ConvAlgo.Native "
                "(implicit gemm off)",
                flush=True,
            )

    @HOOKS.register_module()
    class GeoProgress(HookBase):
        def before_train(self):
            # Installing the proxy is also the enable switch: Pointcept skips
            # every scalar when ``writer`` is None, so there is no separate
            # "logging on" flag to set and no tensorboard files to write.
            self.trainer.writer = _MlflowWriter()

        def after_epoch(self):
            epoch = self.trainer.epoch + 1
            print(f"-> epoch {epoch}/{self.trainer.max_epoch}", flush=True)

    @TRAINERS.register_module("GeoTrainer")
    class GeoTrainer(Trainer):
        """``Trainer`` that can stop cleanly between epochs.

        The loop is copied from ``Trainer.train`` rather than wrapped because
        ``ExceptionWriter`` converts any exception raised inside it into
        ``sys.exit(1)``: raising to break out would report a crash and skip
        ``after_train``, losing the model. Lowering ``max_epoch`` does not work
        either -- ``range()`` is evaluated once, when the loop starts.
        """

        def train(self):
            with EventStorage() as self.storage, ExceptionWriter():
                self.before_train()
                self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        self.before_step()
                        self.run_step()
                        self.after_step()
                    self.after_epoch()
                    if stop_requested:
                        self.logger.info("=> stop requested; ending after this epoch")
                        break
                self.after_train()

    return GeoProgress, GeoTrainer


# --- pretrained weights ------------------------------------------------------


def _md5(path: Path) -> str:
    import hashlib

    digest = hashlib.md5()  # noqa: S324 - integrity, not a security boundary
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_weight(config: dict) -> "Path | None":
    """The public checkpoint for this model variant, fetching it if needed.

    The stager normally has it in place already (plan 37). A mirror miss is not
    a failure -- it is slower: the file is downloaded here, exactly as the
    other frameworks' own loaders would have. ``None`` means this variant ships
    no public checkpoint, so the run starts from scratch.
    """

    asset = MODEL_WEIGHTS.get(str(config.get("model") or ""))
    if asset is None:
        return None
    path = WEIGHTS_DIR / asset["filename"]
    if path.is_file() and _md5(path) == asset["md5"]:
        print(f"-> pretrained: {path} (staged)", flush=True)
        return path

    print(f"-> pretrained: not staged; fetching {asset['url']}", flush=True)
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(asset["url"], partial)  # noqa: S310 - fixed URL
        actual = _md5(partial)
        if actual != asset["md5"]:
            raise ValueError(f"md5 {actual} != {asset['md5']}")
        partial.replace(path)
    except Exception as exc:  # noqa: BLE001
        partial.unlink(missing_ok=True)
        # Not fatal: the catalog checkpoint is an accelerator, and a run from
        # scratch is a worse model, not a wrong one. A *tenant's* checkpoint is
        # the opposite -- see ``resolve_weight``.
        print(
            f"-> pretrained download failed ({exc}); training from scratch", flush=True
        )
        return None
    return path


def resolve_weight(config: dict) -> "Path | None":
    """The checkpoint this run starts from, before filtering."""

    pretrained = config.get("pretrained") or {}
    if pretrained.get("source") == "tenant":
        path = Path(str(pretrained.get("path") or ""))
        if not path.is_file():
            # Pointcept would log "No weight found" and train from scratch, and
            # the run would report success having ignored the checkpoint the
            # submitter picked (plan 36 Phase 0).
            raise SystemExit(f"pretrained weights not staged: {str(path)!r}")
        print(f"-> pretrained: {pretrained.get('name') or path}", flush=True)
        return path
    return catalog_weight(config)


def adapt_weight(src: Path, dst: Path, model) -> dict:
    """Rewrite a checkpoint so Pointcept's loader accepts it.

    ``CheckpointLoader`` calls ``load_state_dict(..., strict=False)``, and
    ``strict=False`` forgives a missing key but **not** a shape mismatch: a
    checkpoint trained on 20 classes stops a 3-class run dead in
    ``seg_head.weight``. Re-using a public backbone with your own label set is
    the whole point of starting from one, so the head is dropped and
    re-initialised, and everything dropped is named in the log.

    A checkpoint that overlaps the model too little is refused instead. That is
    not a head mismatch but a different architecture, and training from an
    almost-empty load would look like fine-tuning while being scratch.
    """

    import torch

    raw = torch.load(src, map_location="cpu", weights_only=False)
    source = raw.get("state_dict", raw)
    target = model.state_dict()

    kept, dropped = {}, []
    for key, value in source.items():
        name = key[7:] if key.startswith("module.") else key
        if name in target and tuple(target[name].shape) == tuple(value.shape):
            kept[name] = value
        else:
            dropped.append(name)

    ratio = len(kept) / max(1, len(target))
    if ratio < 0.5:
        raise SystemExit(
            f"pretrained checkpoint matches only {len(kept)}/{len(target)} "
            f"tensors ({ratio:.0%}) -- a different architecture, not a head "
            "mismatch"
        )
    if dropped:
        shown = ", ".join(sorted(dropped)[:6])
        tail = " ..." if len(dropped) > 6 else ""
        print(
            f"-> pretrained: re-initialising {len(dropped)} tensor(s): {shown}{tail}",
            flush=True,
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": kept}, dst)
    return {"kept": len(kept), "dropped": len(dropped), "total": len(target)}


# --- model logging -----------------------------------------------------------


def best_checkpoint() -> "Path | None":
    """The checkpoint to serve: best on validation, else the last one written.

    A run stopped at an epoch boundary has both; a run stopped before its first
    evaluation finished has only ``model_last``. Preferring best and falling
    back is what keeps an interrupted run from losing its model entirely.
    """

    model_dir = OUTPUT_DIR / "model"
    for name in ("model_best.pth", "model_last.pth"):
        path = model_dir / name
        if path.is_file():
            return path
    return None


def _pip_requirements() -> list:
    """Serving-image requirements — the pip-installable part of this runtime.

    It is deliberately **not** the whole truth. Pointcept is a source tree with
    no published package, and its native wheels (spconv, pointops, flash-attn,
    the PyG companions) are compiled for one torch × CUDA × arch combination by
    this image's own builder stage. Nothing pip can reach reproduces that, so a
    serving image is built from the training image rather than from this list;
    ``geo_runtime_image`` in the metadata records which one. Serving a Pointcept
    model end to end is a separate plan (plan 36 §7).
    """

    import torch

    torch_version = torch.__version__.split("+")[0]
    return [f"torch=={torch_version}", "numpy", "plyfile"]


def _serving_signature():
    """A base64 PLY in, the voxel size as the one knob (plan 18 §1.1).

    **No ``conf``**: semantic segmentation assigns every point its argmax
    class. There is no score threshold to raise without inventing an "unknown"
    class the model was never trained to predict.
    """

    import pointcept_pyfunc

    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

    return ModelSignature(
        inputs=Schema([ColSpec(DataType.string, pointcept_pyfunc.CLOUD_COLUMN)]),
        params=ParamSchema(
            [
                ParamSpec("grid_size", DataType.double, 0.0),
                ParamSpec(
                    "include_ply",
                    DataType.boolean,
                    pointcept_pyfunc.DEFAULT_INCLUDE_PLY,
                ),
            ]
        ),
    )


def _serving_metadata(config: dict, params: dict) -> dict:
    """MLmodel ``metadata`` — what the serving console and builder read back."""

    classes = list(config.get("classes") or [])
    return {
        "geo_task": config.get("task", ""),
        "geo_framework": config.get("framework", "pointcept"),
        "input_kind": "pointcloud_b64",
        "class_names": {str(i): name for i, name in enumerate(classes)},
        "serving_runtime": "gpu",
        "apt_packages": list(SERVING_APT_PACKAGES),
        # The image that trained this model. Its wheels are the only build of
        # spconv/pointops/flash-attn that matches the checkpoint's torch, so a
        # serving image has to start from here rather than from pip.
        "geo_runtime_image": os.environ.get("GEO_JOB_IMAGE", ""),
        "grid_size": float(params["grid_size"]),
    }


def model_config(config: dict, params: dict, has_color: bool) -> dict:
    """What ``pointcept_pyfunc`` needs to rebuild the model at serving time."""

    return {
        "task": config.get("task", ""),
        "variant": config.get("model", ""),
        "class_names": list(config.get("classes") or []),
        "grid_size": float(params["grid_size"]),
        "has_color": bool(has_color),
    }


def log_model(config: dict, params: dict, has_color: bool) -> None:
    """Log the trained checkpoint as an MLflow **pyfunc** model (contract §5)."""

    run_id = _run_id()
    if not run_id or not os.environ.get("MLFLOW_TRACKING_URI"):
        print("-> no MLflow run/tracking; skipping model logging", flush=True)
        return

    import mlflow

    weights = best_checkpoint()
    if weights is None:
        raise FileNotFoundError("no trained checkpoint to log")
    generated = OUTPUT_DIR / "config.py"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # MLflow 3 copies a ``uv.lock``+``pyproject.toml`` found above cwd into the
    # model artifact; that is the platform's lock, not this model's.
    os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")

    artifacts = {"weights": str(weights)}
    if generated.is_file():
        # The config is not a nicety: rebuilding the network at serving time
        # needs the exact architecture this run trained, and the hyperparameter
        # whitelist can change under a saved model.
        artifacts["config"] = str(generated)

    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=str(Path(__file__).with_name("pointcept_pyfunc.py")),
            artifacts=artifacts,
            signature=_serving_signature(),
            metadata=_serving_metadata(config, params),
            model_config=model_config(config, params, has_color),
            pip_requirements=_pip_requirements(),
        )
    print(f"-> logged model {info.model_uri}", flush=True)


# --- main --------------------------------------------------------------------


def train(config: dict, data_root: Path, has_color: bool) -> dict:
    """Build the config, run Pointcept, return the hyperparameters used."""

    params = _hyperparameters(config)
    classes = [str(c) for c in (config.get("classes") or [])]
    if not classes:
        raise ValueError("no classes selected; semantic segmentation needs at least 1")

    client = _client()
    run_id = _run_id()
    register_extensions(client, run_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = WORKDIR / "pointcept_config.py"
    config_path.write_text(
        build_config_text(
            data_root, OUTPUT_DIR, classes, params, has_color=has_color, weight=""
        ),
        encoding="utf-8",
    )

    from pointcept.engines.defaults import default_config_parser, default_setup
    from pointcept.engines.train import TRAINERS

    cfg = default_config_parser(str(config_path), None)
    cfg = default_setup(cfg)

    source = resolve_weight(config)
    if source is not None:
        from pointcept.models import build_model

        probe = build_model(cfg.model)
        adapted = WEIGHTS_DIR / "start.pth"
        print(
            f"-> pretrained adapted: {adapt_weight(source, adapted, probe)}", flush=True
        )
        del probe
        cfg.weight = str(adapted)

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()
    if stop_requested:
        raise StopTraining("stopped at an epoch boundary")
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    # Kept for compatibility with older launchers; the container runs this file
    # directly and the id comes from ``experiment.json``.
    parser.add_argument("--experiment-id", required=False, default="")
    parser.parse_args()

    install_sigterm_flag()

    config_path = Path(os.environ.get("GEO_CONFIG") or (WORKDIR / "experiment.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"-> training {config['name']} ({config['experiment_id']})", flush=True)

    params = _hyperparameters(config)
    # Held outside the try so the stop path can still describe the run: whether
    # the dataset had colour decides the model's input channels, and guessing
    # would log a model that cannot be rebuilt.
    has_color = True
    try:
        clouds = annotations.load_pointclouds(DATA_DIR)
        if not clouds:
            raise ValueError(
                "no point clouds in the staged dataset; this framework reads .ply"
            )
        data_root, has_color = build_dataset(
            clouds, config.get("classes") or [], config.get("split") or {}
        )
        params = train(config, data_root, has_color)
        for name in ("train.log", "config.py"):
            candidate = OUTPUT_DIR / name
            if candidate.is_file():
                _log_run_artifact(candidate)
        log_model(config, params, has_color)
        print("-> training finished", flush=True)
    except StopTraining as exc:
        # A stop after an epoch finished still has a model: the checkpoint
        # saver wrote one at that boundary. A stop during data preparation has
        # nothing, and ``log_model`` says so rather than inventing one. Either
        # way the server reports STOPPED regardless of the exit code (§4).
        print(f"-> {exc}", flush=True)
        try:
            log_model(config, params, has_color)
        except Exception:  # noqa: BLE001 - a stop must not turn into a failure
            traceback.print_exc()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

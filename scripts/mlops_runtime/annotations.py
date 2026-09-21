# -*- coding: utf-8 -*-

"""Staged-dataset reading and annotation parsing — shared by every trainer.

Each framework wants its labels in its own layout (YOLO text files, COCO JSON,
...), but they all start from the same place: the files the platform staged
into ``GEO_DATA_DIR`` and the three annotation formats the pipeline stores
(COCO, LabelMe JSON, VOC XML). Copying that reader into every trainer means a
parser fix lands in one image and not the other, so it lives here and the
Dockerfiles ``COPY`` it next to the entrypoint they build (plan 34).

**stdlib only.** These helpers run before any framework is imported, and the
unit tests exercise them without torch installed -- an import of numpy or PIL
here would make both of those false.
"""

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

#: Scratch space. ``GEO_WORK_DIR`` is the contract; cwd is the fallback for a
#: local dry run without the variable.
WORKDIR = Path(os.environ.get("GEO_WORK_DIR") or Path.cwd())

#: Read-only staged dataset (``manifest.json`` + files, trainer contract §1.1).
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR") or (WORKDIR / "dataset"))


def load_dataset(data_dir: Path | None = None) -> tuple[list[Path], list[Path]]:
    """Read the staged dataset from ``GEO_DATA_DIR`` (no download).

    The platform put the files there before this process started, so there is
    nothing to fetch and no storage credential in this environment. Falls back
    to walking the directory when ``manifest.json`` is missing, since the files
    are what actually matter and refusing over a missing index would strand a
    perfectly usable directory.
    """

    root = data_dir or DATA_DIR
    manifest_path = root / "manifest.json"
    images: list[Path] = []
    annotations: list[Path] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("files", []):
            path = root / record["path"]
            if record.get("kind") == "image":
                images.append(path)
            elif record.get("kind") == "annotation":
                annotations.append(path)
    else:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            if path.suffix.lower() in (".json", ".xml"):
                annotations.append(path)
            else:
                images.append(path)
    print(
        f"-> dataset: {len(images)} image(s), {len(annotations)} annotation(s)",
        flush=True,
    )
    return images, annotations


def load_pointclouds(data_dir: Path | None = None) -> list[Path]:
    """The staged ``pointcloud`` files (plan 36).

    A separate reader rather than a third return value from
    :func:`load_dataset`: point clouds carry their labels *inside* the file, so
    a trainer that wants them wants no annotation files at all, and widening
    the existing tuple would change a signature two other trainers depend on.

    Same manifest-first, walk-as-fallback rule as :func:`load_dataset`, for the
    same reason -- the files are what matter, and a missing index should not
    strand a directory that is perfectly usable.
    """

    root = data_dir or DATA_DIR
    manifest_path = root / "manifest.json"
    clouds: list[Path] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("files", []):
            if record.get("kind") == "pointcloud":
                clouds.append(root / record["path"])
    else:
        clouds = [p for p in sorted(root.rglob("*.ply")) if p.is_file()]
    print(f"-> dataset: {len(clouds)} point cloud(s)", flush=True)
    return sorted(clouds)


# --- annotation conversion (COCO / LabelMe JSON / VOC XML -> YOLO bboxes) -----


def _points_to_bbox(points) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _coco_polygon(segmentation) -> list | None:
    """First polygon of a COCO ``segmentation`` as ``[(x, y), ...]``.

    Returns ``None`` for RLE masks (a dict) or degenerate rings — the caller
    then falls back to the bbox rectangle for segmentation labels.
    """

    if not isinstance(segmentation, list) or not segmentation:
        return None
    poly = segmentation[0]
    if not isinstance(poly, list) or len(poly) < 6:  # need >= 3 (x, y) pairs
        return None
    return [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]


def _parse_coco(data: dict, boxes: dict) -> None:
    images = {
        img["id"]: (
            Path(img.get("file_name", "")).stem,
            img.get("width"),
            img.get("height"),
        )
        for img in data.get("images", [])
    }
    names = {c.get("id"): c.get("name") for c in data.get("categories", [])}
    for ann in data.get("annotations", []):
        stem, width, height = images.get(ann.get("image_id"), ("", None, None))
        label = names.get(ann.get("category_id"))
        bbox = ann.get("bbox")
        if not stem or label is None or not bbox or not width or not height:
            continue
        x, y, w, h = (float(v) for v in bbox)
        points = _coco_polygon(ann.get("segmentation"))
        boxes.setdefault(stem, []).append(
            (label, x, y, x + w, y + h, width, height, points)
        )


def _declared_stem(value) -> str:
    """Stem of a path an annotation file declares for its image.

    Backslashes are folded to ``/`` first: a LabelMe/VOC file written on
    Windows carries ``..\\images\\img_001.jpg``, which POSIX ``Path`` reads as
    one long filename and would never match a staged image. The stager
    (``training_stager/stage.py``) flattens stored names the same way.
    """

    return Path(str(value or "").replace("\\", "/")).stem


def _resolve_stem(declared, file_stem: str, image_stems: set | None) -> str:
    """The image stem an annotation belongs to.

    The declared image (LabelMe ``imagePath``, VOC ``<filename>``) wins when it
    names an image that was actually staged -- that is the pairing the labeller
    recorded, and it survives an annotation file being renamed. It is only
    trusted against ``image_stems`` because the declared name can be a stale
    pre-dedup one that matches no downloaded image; keying on it blindly leaves
    every label file empty and validation reporting no labels.

    Everything else falls back to the annotation FILE's own stem, which is how
    the pipeline stores each image/annotation pair. That includes the case
    where the caller passes no ``image_stems`` at all: with nothing to match
    against, "the declared name matches" is not a claim that can be made.
    """

    stem = _declared_stem(declared)
    if stem and image_stems is not None and stem in image_stems:
        return stem
    return file_stem or stem


def _parse_labelme(
    data: dict, boxes: dict, file_stem: str = "", image_stems: set | None = None
) -> None:
    stem = _resolve_stem(data.get("imagePath"), file_stem, image_stems)
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not stem:
        return
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points") or []
        if label is None or len(points) < 2:
            continue
        x1, y1, x2, y2 = _points_to_bbox(points)
        # A 2-point LabelMe shape is a rectangle (bbox only); >= 3 is a polygon.
        poly = (
            [(float(p[0]), float(p[1])) for p in points] if len(points) >= 3 else None
        )
        boxes.setdefault(stem, []).append((label, x1, y1, x2, y2, width, height, poly))


def _parse_voc(
    root: ET.Element, boxes: dict, file_stem: str = "", image_stems: set | None = None
) -> None:
    # Same rule as LabelMe: the XML's ``<filename>`` wins when it names a staged
    # image, else the annotation FILE's stem.
    stem = _resolve_stem(root.findtext("filename"), file_stem, image_stems)
    size = root.find("size")

    def _dimension(key: str) -> int | None:
        """``<size><width>`` 등. 없거나 숫자가 아니면 ``None``.

        예전에는 곧바로 ``int()`` 에 넣어, ``<size>`` 가 비거나 값이 비숫자인 VOC
        파일 하나가 학습 전체를 TypeError/ValueError 로 세웠다. 좌표 정규화에 쓰이는
        값이라 없으면 그 파일을 건너뛰는 것이 맞다.
        """

        if size is None:
            return None
        text = (size.findtext(key) or "").strip()
        try:
            return int(text)
        except ValueError:
            return None

    width = _dimension("width")
    height = _dimension("height")
    if not stem:
        return
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        polygon = obj.find("polygon")
        poly = None
        if polygon is not None:
            points = [
                (float(pt.findtext("x") or 0), float(pt.findtext("y") or 0))
                for pt in polygon.findall("pt")
            ]
            if not points:
                continue
            x1, y1, x2, y2 = _points_to_bbox(points)
            if len(points) >= 3:
                poly = points
        else:
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            x1, y1, x2, y2 = (
                float(bnd.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax")
            )
        boxes.setdefault(stem, []).append((label, x1, y1, x2, y2, width, height, poly))


def parse_annotations(
    annotation_paths: list[Path], images: list[Path] | None = None
) -> dict:
    """stem -> [(label, x1, y1, x2, y2, img_w|None, img_h|None, points|None)].

    The stem is matched in :func:`build_yolo_dataset` against each image's
    basename. For COCO it is the ``file_name`` stem. For the per-image formats
    (LabelMe/VOC) it is the image the file declares — ``imagePath`` /
    ``<filename>`` — when that names one of ``images``, else the annotation
    FILE's own basename; see :func:`_resolve_stem`. Pass ``images`` (the staged
    images from :func:`load_dataset`) to get that preference: without it there
    is nothing to match a declared name against, so the file's own stem is used.

    ``points`` is the source polygon (``[(x, y), ...]``, pixel coords) when the
    annotation carries one, else ``None`` — segmentation label writing keeps it,
    detection ignores it.
    """

    image_stems = {p.stem for p in images} if images is not None else None
    boxes: dict = {}
    for path in annotation_paths:
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if "annotations" in data and "images" in data:
                    _parse_coco(data, boxes)
                elif "shapes" in data:
                    _parse_labelme(data, boxes, path.stem, image_stems)
            elif path.suffix.lower() == ".xml":
                _parse_voc(ET.parse(path).getroot(), boxes, path.stem, image_stems)
        except Exception as exc:  # noqa: BLE001 - skip broken files, keep rest
            print(f"-> skipping annotation {path.name}: {exc}", flush=True)
    return boxes


# --- train/val split ---------------------------------------------------------


def split_images(images: list[Path], split: dict) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val split of the staged images.

    Shared so two frameworks trained on the same dataset and seed see the same
    partition -- otherwise their metrics are not comparable and a "which model
    is better" answer is partly an artefact of the split.

    Always keeps at least one image on each side when there is more than one:
    a validation pass over nothing yields no metrics at all, which reads as a
    broken run rather than a small dataset.
    """

    ordered = sorted(images, key=lambda p: p.name)
    random.Random(int(split.get("seed", 0))).shuffle(ordered)
    train_count = max(
        1, round(len(ordered) * int(split.get("train_percent", 80)) / 100)
    )
    if len(ordered) > 1:
        train_count = min(train_count, len(ordered) - 1)
    return ordered[:train_count], ordered[train_count:]

# -*- coding: utf-8 -*-

"""Pointcept 서빙 wrapper — MLflow "models from code" 진입점 (plan 36).

학습이 남긴 체크포인트를 **MLflow pyfunc 모델**로 감싼다. 응답은 이미지 트레이너
둘과 **다른 모양**이다 — 같을 수가 없다. 탐지·인스턴스 분할은 물체 몇 개를
돌려주지만, 시맨틱 세그멘테이션은 **점 하나마다** 클래스를 매긴다::

    POST /invocations
    {"dataframe_split": {"columns": ["pointcloud_b64"], "data": [["cGx5CmZvcm..."]]},
     "params": {"grid_size": 0.6, "include_ply": true}}

    {"predictions": [{"task": "semantic_segmentation", "point_count": 24000,
      "classes": [{"cls": 0, "name": "0", "points": 15745, "ratio": 0.656}, ...],
      "ply_b64": "cGx5CmZvcm..."}]}

점별 라벨 배열을 그대로 JSON 에 실으면 백만 개짜리 정수 목록이 된다. 대신
**클래스별 집계**(사람이 읽는 답)와 **라벨 색이 칠해진 PLY**(뷰어가 읽는 답)를
돌려준다 — 프론트에 이미 `.ply` 뷰어가 있다.

``conf`` 파라미터는 **없다**. 세그멘테이션은 모든 점에 argmax 클래스를 준다.
임계값을 두려면 모델이 배운 적 없는 "미분류" 클래스를 발명해야 한다.

**이 파일은 ``geo_mlops`` 를 import 하지 않는다.** MLflow 가 이 스크립트 하나만
아티팩트에 복사하므로 서빙 컨테이너 안에서 홀로 실행될 수 있어야 하고, 무거운
의존성(torch/pointcept/numpy)은 전부 함수 안에서 import 한다 — 그래야 그것들이
없는 환경에서도 모듈을 import 해 순수 함수를 단위 테스트할 수 있다.

**서빙 이미지는 pip 로 만들 수 없다.** Pointcept 은 배포된 패키지가 아니고, 네이티브
휠(spconv·pointops·flash-attn)은 학습 이미지의 빌더 스테이지가 한 torch × CUDA ×
arch 조합에 맞춰 구운 것이다. MLmodel 의 ``geo_runtime_image`` 가 그 이미지를
가리키며, 실제 배포는 후속 플랜 소관이다(plan 36 §7).
"""

import base64
import binascii
import io
from typing import Any, Optional

import pandas as pd

import mlflow
from mlflow.pyfunc import PythonModel

#: 요청 DataFrame 이 담는 유일한 필수 컬럼 (base64 PLY).
CLOUD_COLUMN = "pointcloud_b64"

#: ``params`` 기본값. ``grid_size`` 0 은 "학습 때 값을 쓰라"는 뜻이다 — 서명에
#: 학습값을 박아 두면 모델마다 기본값이 달라져 콘솔이 무엇을 보여줄지 모른다.
DEFAULT_GRID_SIZE = 0.0
DEFAULT_INCLUDE_PLY = True

#: 돌려주는 PLY 의 최대 점 수. 원본이 수백만 점이면 응답이 수십 MB 가 되는데,
#: 뷰어에는 그만한 해상도가 필요 없다. 넘으면 등간격으로 솎는다 — 라벨 분포는
#: 유지되고, 집계 수치는 **솎기 전 전체**로 낸다.
MAX_PLY_POINTS = 200_000

#: 클래스별 색 (dataset_files.LABEL_PALETTE 와 같은 20색). 프리뷰와 추론 결과가
#: 같은 색이면 사용자가 둘을 눈으로 맞춰 볼 수 있다.
LABEL_PALETTE = (
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
)


def decode_cloud(value: Any) -> bytes:
    """base64 문자열(또는 data URL)을 PLY 바이트로."""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str):
        raise ValueError(f"{CLOUD_COLUMN} must be a base64 string")
    text = value.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{CLOUD_COLUMN} is not valid base64: {exc}") from exc


def _hex_to_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _cloud_column(frame: pd.DataFrame) -> str:
    """요청이 실제로 쓴 컬럼 이름 (하나뿐이면 이름을 묻지 않는다)."""

    if CLOUD_COLUMN in frame.columns:
        return CLOUD_COLUMN
    if len(frame.columns) == 1:
        return str(frame.columns[0])
    raise ValueError(f"missing column '{CLOUD_COLUMN}'")


def _param(params: Optional[dict], key: str, default):
    if not params:
        return default
    value = params.get(key)
    return default if value is None else value


def summarize(labels, class_names: list) -> list:
    """점별 라벨 배열을 클래스별 집계 행으로.

    학습에 쓰인 **모든** 클래스가 한 행씩 나온다 — 예측이 0 개인 클래스를 빼면
    응답의 행 수가 입력마다 달라져서, 화면이 표를 안정적으로 그릴 수 없다.
    """

    import numpy as np

    labels = np.asarray(labels)
    total = int(labels.size)
    counts = (
        np.bincount(labels, minlength=len(class_names))
        if total
        else np.zeros(len(class_names), dtype=int)
    )
    rows = []
    for index, name in enumerate(class_names):
        points = int(counts[index])
        rows.append(
            {
                "cls": index,
                "name": str(name),
                "points": points,
                "ratio": round(points / total, 6) if total else 0.0,
                "color": LABEL_PALETTE[index % len(LABEL_PALETTE)],
            }
        )
    return rows


def encode_ply(coord, labels, max_points: int = MAX_PLY_POINTS) -> str:
    """라벨 색을 칠한 이진 PLY 를 base64 문자열로."""

    import numpy as np
    from plyfile import PlyData, PlyElement

    coord = np.asarray(coord)
    labels = np.asarray(labels)
    if len(coord) > max_points:
        index = np.linspace(0, len(coord) - 1, max_points).astype(np.int64)
        coord, labels = coord[index], labels[index]

    palette = np.array([_hex_to_rgb(c) for c in LABEL_PALETTE], dtype="u1")
    colors = palette[np.clip(labels, 0, None) % len(LABEL_PALETTE)]

    vertex = np.empty(
        len(coord),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    vertex["label"] = labels.astype("i4")

    buffer = io.BytesIO()
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(
        buffer
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class PointceptSegmentor(PythonModel):
    """MLflow pyfunc — 학습이 남긴 PTv3 체크포인트로 점별 분할을 수행한다."""

    def load_context(self, context) -> None:  # noqa: ANN001 - MLflow 계약
        import torch
        from pointcept.models import build_model
        from pointcept.utils.config import Config

        config = getattr(context, "model_config", None) or {}
        self.class_names = list(config.get("class_names") or [])
        self.grid_size = float(config.get("grid_size") or 0.05)
        self.has_color = bool(config.get("has_color", True))

        # 학습이 생성한 컨피그가 아키텍처의 유일한 진실이다 — 하이퍼파라미터
        # 화이트리스트는 저장된 모델 밑에서 바뀔 수 있고, 그러면 여기서 만든
        # 네트워크가 체크포인트와 어긋난다.
        cfg = Config.fromfile(context.artifacts["config"])
        model = build_model(cfg.model)
        state = torch.load(
            context.artifacts["weights"], map_location="cpu", weights_only=False
        )
        weights = state.get("state_dict", state)
        # DDP 로 저장된 체크포인트는 ``module.`` 접두사를 달고 있고 평범한 모델은
        # 아니다 — 여기서 벗겨 두면 로더가 하나로 끝난다.
        weights = {k.replace("module.", "", 1): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def _segment(self, coord, color, grid_size: float):
        """점별 클래스 id.

        복셀화를 **한 번만** 하고 ``inverse`` 로 원본 점에 되돌린다. Pointcept 의
        ``SemSegTester`` 는 증강을 바꿔 가며 여러 번 훑어 리더보드 숫자를 만드는
        물건이라, 추론 콘솔이 물어보는 것과 다르다.
        """

        import numpy as np
        import torch
        from pointcept.datasets.transform import Compose

        feat_keys = ("coord", "color") if self.has_color else ("coord",)
        pipeline = [
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
        ]
        if self.has_color:
            pipeline.append(dict(type="NormalizeColor"))
        pipeline += [
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "inverse"),
                feat_keys=feat_keys,
            ),
        ]

        data = {"coord": np.asarray(coord, dtype=np.float32)}
        if self.has_color:
            if color is None:
                # 색으로 학습한 모델에 색 없는 구름이 오면 채널 수가 안 맞는다.
                # 거절하는 편이 회색으로 채워 조용히 틀린 답을 주는 것보다 낫다.
                raise ValueError(
                    "this model was trained with colour; the cloud has no RGB"
                )
            data["color"] = np.asarray(color, dtype=np.float32)
        data["segment"] = np.zeros(len(data["coord"]), dtype=np.int32)

        out = Compose(pipeline)(data)
        inverse = out.pop("inverse")
        batch = {
            key: value.to(self.device)
            for key, value in out.items()
            if isinstance(value, torch.Tensor)
        }
        batch["offset"] = torch.tensor([len(batch["coord"])], device=self.device)
        with torch.inference_mode():
            logits = self.model(batch)["seg_logits"]
        voxel = logits.argmax(dim=1).cpu().numpy()
        return voxel[np.asarray(inverse)]

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ANN001
        import numpy as np
        from plyfile import PlyData

        column = _cloud_column(model_input)
        grid_size = float(_param(params, "grid_size", DEFAULT_GRID_SIZE))
        if grid_size <= 0:
            grid_size = self.grid_size
        include_ply = bool(_param(params, "include_ply", DEFAULT_INCLUDE_PLY))

        predictions = []
        for value in model_input[column].tolist():
            vertex = PlyData.read(io.BytesIO(decode_cloud(value)))["vertex"].data
            names = list(vertex.dtype.names or ())
            for axis in ("x", "y", "z"):
                if axis not in names:
                    raise ValueError(f"PLY vertex has no '{axis}' property")
            coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(
                np.float32
            )
            color = None
            if {"red", "green", "blue"}.issubset(names):
                color = np.stack(
                    [vertex["red"], vertex["green"], vertex["blue"]], axis=1
                ).astype(np.float32)

            labels = self._segment(coord, color, grid_size)
            item = {
                "task": "semantic_segmentation",
                "point_count": int(len(labels)),
                "grid_size": grid_size,
                "classes": summarize(labels, self.class_names),
            }
            if include_ply:
                item["ply_b64"] = encode_ply(coord, labels)
            predictions.append(item)
        return {"predictions": predictions}


mlflow.models.set_model(PointceptSegmentor())

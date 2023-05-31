from itertools import product

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from nos import hub
from nos.common.spec import FunctionSignature, ModelSpec
from nos.common.tasks import TaskType
from nos.common.types import Batch, EmbeddingSpec, ImageSpec, ImageT, TensorSpec, TensorT


EMBEDDING_SHAPES = [
    (512,),
    (512,),
]
IMAGE_SHAPES = [
    (224, 224, 3),
    (224, 224, 3),
    (None, None, 3),
    (None, None, 3),
]
TENSOR_SHAPES = [
    *EMBEDDING_SHAPES,
    (1, 4, 4),
    (None, 4, 4),
    *IMAGE_SHAPES,
]
DTYPES = ["uint8", "int32", "int64", "float32", "float64"]


def test_common_tensor_spec_valid_shapes():
    """Test valid tensor shapes."""
    for shape, dtype in product(TENSOR_SHAPES, DTYPES):
        spec = TensorSpec(shape=shape, dtype=dtype)
    assert spec is not None


def test_common_tensor_spec_invalid_shapes():
    """Test invalid tensor shapes."""
    INVALID_SHAPES = [
        (1, 224, 224, 3, 3),
        (None, 224, 224, 3, 3),
        (1, 224, 224, 1, 2, 3),
        (None, 224, 224, 1, 2, 3),
        (None, None, None, None, None),
    ]
    for shape in INVALID_SHAPES:
        with pytest.raises(ValueError):
            spec = TensorSpec(shape=shape, dtype="uint8")
            assert spec is not None


def test_common_image_spec_valid_shapes():
    """Test valid image shapes."""
    for shape, dtype in product(IMAGE_SHAPES, DTYPES):
        spec = ImageSpec(shape=shape, dtype=dtype)
        assert spec is not None


def test_common_embedding_spec_valid_shapes():
    """Test valid embedding shapes."""
    for shape, dtype in product(EMBEDDING_SHAPES, DTYPES):
        spec = EmbeddingSpec(shape=shape, dtype=dtype)
        assert spec is not None


@pytest.fixture
def img2vec_signature():
    yield FunctionSignature(
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )


def test_common_model_spec(img2vec_signature):
    """Test model spec."""
    from typing import Union

    import numpy as np
    from PIL import Image

    class TestImg2VecModel:
        def __init__(self, model_name: str = "openai/clip"):
            self.model = hub.load(model_name)

        def __call__(self, img: Union[Image.Image, np.ndarray]) -> np.ndarray:
            embedding = self.model(img)
            return embedding

    spec = ModelSpec(
        name="openai/clip",
        task=TaskType.IMAGE_EMBEDDING,
        signature=FunctionSignature(
            inputs=img2vec_signature.inputs,
            outputs=img2vec_signature.outputs,
            func_or_cls=TestImg2VecModel,
            init_args=("openai/clip",),
            init_kwargs={},
            method_name="__call__",
        ),
    )
    assert spec is not None

    # Create a model spec with a wrong method name
    with pytest.raises(ValidationError):
        spec = ModelSpec(
            name="openai/clip",
            task=TaskType.IMAGE_EMBEDDING,
            signature=FunctionSignature(
                inputs=img2vec_signature.inputs,
                outputs=img2vec_signature.outputs,
                func_or_cls=TestImg2VecModel,
                init_args=("openai/clip",),
                init_kwargs={},
                method_name="predict",
            ),
        )
        assert spec is not None


def test_common_model_spec_variations():
    # Create signatures for all tasks (without func_or_cls, init_args, init_kwargs, method_name)
    ImageSpec(shape=(None, None, 3), dtype="uint8")

    # Image embedding (img2vec)
    img2vec_signature = FunctionSignature(
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )
    assert img2vec_signature is not None

    # Text embedding (txt2vec)
    txt2vec_signature = FunctionSignature(
        inputs={"texts": str},
        outputs={"embedding": Batch[TensorT[np.ndarray, EmbeddingSpec(shape=(512,), dtype="float32")]]},
    )
    assert txt2vec_signature is not None

    # Object detection (img2bbox)
    img2bbox_signature = FunctionSignature(
        inputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
        outputs={
            "scores": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "labels": Batch[TensorT[np.ndarray, TensorSpec(shape=(None), dtype="float32")]],
            "bboxes": Batch[TensorT[np.ndarray, TensorSpec(shape=(None, 4), dtype="float32")]],
        },
    )
    assert img2bbox_signature is not None

    # Image generation (txt2img)
    txt2img_signature = FunctionSignature(
        inputs={"texts": Batch[str]},
        outputs={"images": Batch[ImageT[Image.Image, ImageSpec(shape=(None, None, 3), dtype="uint8")]]},
    )
    assert txt2img_signature is not None


def check_object_type(v):
    if isinstance(v, list):
        for item in v:
            check_object_type(item)
        return

    assert isinstance(v.is_batched(), bool)
    if v.batch_size():
        assert isinstance(v.batch_size(), int)
    assert v.base_type() is not None
    if v.base_spec():
        spec = v.base_spec()
        assert hasattr(spec, "shape")
        assert hasattr(spec, "dtype")


def test_common_spec_signature():
    """Test function signature."""
    from loguru import logger

    for spec in hub.list():
        logger.debug(f"{spec.name}, {spec.task}")
        assert spec is not None
        assert spec.name == spec.name
        assert spec.task == spec.task
        assert spec.signature.inputs is not None
        assert spec.signature.outputs is not None

        assert isinstance(spec.signature.inputs, dict)
        assert isinstance(spec.signature.outputs, dict)
        logger.debug(f"{spec.name}, {spec.task}")

        for k, v in spec.signature.get_inputs_spec().items():
            logger.debug(f"input: {k}, {v}")
            check_object_type(v)
        for k, v in spec.signature.get_outputs_spec().items():
            logger.debug(f"output: {k}, {v}")
            check_object_type(v)
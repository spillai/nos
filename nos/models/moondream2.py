from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from PIL import Image

from nos.hub import HuggingFaceHubConfig


@dataclass(frozen=True)
class MoonDream2Config(HuggingFaceHubConfig):
    revision: str = "2024-03-05"
    """Revision of the model."""


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class MoonDream2:
    configs = {
        "vikhyatk/moondream2": MoonDream2Config(
            model_name="vikhyatk/moondream2",
            revision="2024-03-05",
        ),
    }

    def __init__(self, model_name: str = "vikhyatk/moondream2"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            self.cfg = MoonDream2.configs[model_name]
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}, available models: {MoonDream2.configs.keys()}")
        model_name = self.cfg.model_name
        self.device = auto_device()

        torch_dtype = torch.float32 if self.device == "cuda" else torch.float16
        print(
            f"Initializing MoonDream2 model (model_name={model_name}, device={self.device}, torch_dtype={torch_dtype})"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, revision=self.cfg.revision
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, revision=self.cfg.revision)

    @torch.inference_mode()
    def _encode_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Encode image into an embedding."""
        return self.model.encode_image(image)

    @torch.inference_mode()
    def __call__(self, image: Union[Image.Image, np.ndarray], text: str) -> np.ndarray:
        """Answer questions based on the image and text."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        emb = self._encode_image(image)
        return self.model.answer_question(emb, text, self.tokenizer)

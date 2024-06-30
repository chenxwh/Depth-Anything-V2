# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path, BaseModel
from depth_anything_v2.dpt import DepthAnythingV2


CMAP = matplotlib.colormaps.get_cmap("Spectral_r")
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
ENCODER2NAME = {
    "vits": "Small",
    "vitb": "Base",
    "vitl": "Large",
    # 'vitg': 'Giant', # we are undergoing company review procedures to release our giant model checkpoint
}

MODEL_URL = (
    "https://weights.replicate.delivery/default/depth-anything/Depth-Anything-V2.tar"
)
MODEL_CACHE = "checkpoints"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class ModelOutput(BaseModel):
    color_depth: Path
    grey_depth: Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.models = {}
        for encoder in ENCODER2NAME.keys():
            self.models[encoder] = DepthAnythingV2(**MODEL_CONFIGS[encoder])
            filepath = f"{MODEL_CACHE}/depth_anything_v2_{encoder}.pth"
            state_dict = torch.load(filepath, map_location="cpu")
            self.models[encoder].load_state_dict(state_dict)
            self.models[encoder] = self.models[encoder].to("cuda").eval()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        model_size: str = Input(
            description="Choose a model",
            choices=list(ENCODER2NAME.values()),
            default="Large",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        model = self.models[f"vit{model_size[0].lower()}"]
        image = cv2.imread(str(image))

        depth = model.infer_image(image[:, :, ::-1])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        grey_depth = Image.fromarray(depth)
        colored_depth = (CMAP(depth)[:, :, :3] * 255).astype(np.uint8)
        colored_depth = Image.fromarray(colored_depth)

        color_out = "/tmp/color.png"
        grey_out = "/tmp/grey.png"
        grey_depth.save(grey_out)
        colored_depth.save(color_out)
        return ModelOutput(color_depth=Path(color_out), grey_depth=Path(grey_out))

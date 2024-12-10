import os
import shutil

import keras
import numpy as np

from decoder import TokenDecoder

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import torch
from strhub.models.utils import load_from_checkpoint
from torchvision import transforms as T
from PIL import Image
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from torch.nn import functional as F
from nobuco.layers.weight import WeightLayer

model = load_from_checkpoint("last.ckpt")
# model = torch.hub.load(
#     "baudm/parseq", "parseq", pretrained=True, refine_iters=0, decode_ar=False
# )
# checkpoint = torch.load("last.ckpt", map_location=torch.device("cpu"))
# model.load_state_dict(checkpoint["state_dict"])
parseq = model.to("cpu").eval()

img_transform = T.Compose(
    [
        T.Resize(parseq.hparams.img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)

# img = Image.open("img.png").convert("RGB") # aga3d
img = Image.open("large.png").convert("RGB")
img: torch.Tensor = img_transform(img).unsqueeze(0)

# parseq.to("cuda").eval()

keras_model = nobuco.pytorch_to_keras(
    parseq,
    args=[img],
    kwargs=None,
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
)

# delete saved_model folder if exists
if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

# export as SavedModel
keras_model.export("saved_model")

np_img: np.ndarray = img.numpy()
img_tensor = tf.convert_to_tensor(np_img)
img_tensor = tf.transpose(img_tensor, perm=(0, 2, 3, 1))

print(img_tensor.shape)

# Make predictions
logits = keras_model(img_tensor)
logits = tf.transpose(logits, perm=(0, 2, 1))
logits = torch.from_numpy(logits.numpy())


print(logits.shape)

outputs = logits.softmax(-1)

token_decoder = TokenDecoder()
pred, conf_scores = token_decoder.decode(outputs)

print(pred[:5], conf_scores[:5])

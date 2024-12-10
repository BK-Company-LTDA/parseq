# from parseq.strhub.models.parseq.model import PARSeq
import torch
from strhub.models.utils import load_from_checkpoint
from torchvision import transforms as T
from PIL import Image

parseq = load_from_checkpoint("last.ckpt", refine_iters=0, decode_ar=False).eval()

# model = torch.hub.load('baudm/parseq', 'parseq')
# checkpoint = torch.load('last.ckpt', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['state_dict'])
# parseq = model.eval()

img_transform = T.Compose(
    [
        T.Resize(parseq.hparams.img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)

img = Image.open("securimage_show (5).png").convert("RGB")
img1 = Image.open("img.png").convert("RGB")

# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img: torch.Tensor = img_transform(img)
img1: torch.Tensor = img_transform(img1)

# imgs_input = torch.stack([img, img1], dim=0)

dummy_input = torch.rand(1, 3, *parseq.hparams.img_size)  # (1, 3, 32, 128) by default

# To ONNX
parseq.to_onnx(
    "parseq.onnx",
    dummy_input,
    opset_version=20,
    do_constant_folding=True,
    export_params=True,
)  # opset v14 or newer is required

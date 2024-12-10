from PIL import Image
import torch
from torchvision import transforms as T
from decoder import TokenDecoder
from strhub.models.utils import load_from_checkpoint

# model = torch.hub.load("baudm/parseq", "parseq")
# checkpoint = torch.load("last.ckpt")
# model.load_state_dict(checkpoint["state_dict"])
# parseq = model.eval()
# parseq = load_from_checkpoint("last.ckpt", map_location=torch.device("cpu"), decode_ar=True).eval()

parseq = torch.load("parseq.pth", map_location=torch.device("cpu")).eval()

img_transform = T.Compose(
    [
        T.Resize(parseq.hparams.img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),
    ]
)

img = Image.open("img.png").convert("RGB")
# img = Image.open("img.png").convert("RGB")

# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img: torch.Tensor = img_transform(img)

# imgs_input = torch.stack([img, img1], dim=0)

logits = parseq(img.unsqueeze(0))
pred = logits.softmax(-1)

decoder = TokenDecoder()

# label, confidence = parseq.tokenizer.decode(pred)

label = decoder.decode(pred)

print(label)

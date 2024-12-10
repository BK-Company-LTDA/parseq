import torch

from parseq.strhub.models.utils import load_from_checkpoint
# from strhub.models.utils import load_from_checkpoint

parseq = load_from_checkpoint("last.ckpt")
torch.save(parseq, "parseq.pth")

# model = torch.load("parseq.pth")
# print(model.model)

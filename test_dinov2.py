import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("Loading DINOv2...")

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14",
    trust_repo=True,
)

model.eval()

x = torch.randn(1, 3, 518, 518)

with torch.no_grad():
    y = model(x)

print("Success!")
print(type(model))
print(y.shape if hasattr(y, "shape") else type(y))

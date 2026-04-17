import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import SimpleCNN

os.makedirs("outputs", exist_ok=True)

transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    "./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
model.eval()

class_names = test_dataset.classes

wrong_images = []
wrong_preds = []
wrong_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        wrong_mask = preds != labels
        wrong_idx = wrong_mask.nonzero(as_tuple=True)[0]

        for idx in wrong_idx:
            wrong_images.append(images[idx].cpu())
            wrong_preds.append(preds[idx].cpu().item())
            wrong_labels.append(labels[idx].cpu().item())

            if len(wrong_images) == 5:
                break

        if len(wrong_images) == 5:
            break

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(wrong_images[i].squeeze(), cmap="gray")
    plt.title(
        f"P:{class_names[wrong_preds[i]]}\nT:{class_names[wrong_labels[i]]}",
        fontsize=9
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/error_samples.png", dpi=200)
plt.show()

print("OKOKOOKK")

for i in range(5):
    print(
        f"Sample {i+1}: "
        f"Pred={class_names[wrong_preds[i]]}, "
        f"True={class_names[wrong_labels[i]]}"
    )
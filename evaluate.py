import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleCNN

transform = transforms.ToTensor()
test_dataset = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))

model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
test_acc = correct / total
print(f"test_acc={test_acc:.4f}")
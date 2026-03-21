from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root='data', transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return loader, dataset.class_to_idx
"""CIFAR-10 data loading, shared across baseline training, watermark
embedding, and any attack that needs real data (e.g. fine-tuning attacks)."""
 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
 
 
def get_cifar_loaders(batch_size=128, data_root="./data", num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
 
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
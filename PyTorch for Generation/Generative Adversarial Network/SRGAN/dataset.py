from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

class SRImageDataset(Dataset):
    def __init__(self, root_dir, hr_size=96, scale_factor=4, train_transform=True):
        self.dataset = datasets.ImageFolder(root_dir)
        self.train = train_transform
        self.lr_size = hr_size // scale_factor

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((hr_size, hr_size)) if self.train else transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        hr_img = self.transform(img)
        lr_img = F.interpolate(hr_img.unsqueeze(0), size=(self.lr_size, self.lr_size)).squeeze(0)
        return hr_img, lr_img
    
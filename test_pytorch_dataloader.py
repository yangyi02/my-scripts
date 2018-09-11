import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

class TestDataset(Dataset):
    def __init__(self, input_size, output_size, transform=None):
        self.input_size = input_size
        self.output_size = output_size
        self.transform = transform

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        image = np.zeros((self.input_size, self.input_size, 3))
        label = np.zeros((self.output_size, self.output_size, 1))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

test_dataset = TestDataset(100, 10)
for i in range(4):
    sample = test_dataset[i]
    print(sample['image'].shape, sample['label'].shape)

transformed_dataset = TestDataset(100, 10, transform=transforms.Compose([ToTensor()]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
for i, sample in enumerate(dataloader):
    print(i, sample['image'].size(), sample['label'].size())

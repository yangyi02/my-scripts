import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time

class TestDataset(Dataset):
    def __init__(self, input_sizes, output_sizes, transform=None):
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        self.transform = transform

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        images = [np.zeros((i_s, i_s, 3)) for i_s in self.input_sizes]
        labels = [np.zeros((o_s, o_s, 1)) for o_s in self.output_sizes]
        sample = {'images': images, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        images = [torch.from_numpy(im.transpose((2, 0, 1))) for im in images]
        labels = [torch.from_numpy(la.transpose((2, 0, 1))) for la in labels]
        return {'images': images,
                'labels': labels}

test_dataset = TestDataset([100, 200], [10, 20])
for i in range(4):
    sample = test_dataset[i]
    print(sample['images'][0].shape, sample['images'][1].shape, sample['labels'][0].shape, sample['labels'][1].shape)

transformed_dataset = TestDataset([100, 200], [10, 20], transform=transforms.Compose([ToTensor()]))

# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)
dataloader = DataLoader(test_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
a = time.time()
for i, sample in enumerate(dataloader):
    print(i, sample['images'][0].size(), sample['images'][1].size(), sample['labels'][0].size(), sample['labels'][1].size())
print(time.time() - a)

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=1)
a = time.time()
for i, sample in enumerate(dataloader):
    print(i, sample['images'][0].size(), sample['images'][1].size(), sample['labels'][0].size(), sample['labels'][1].size())
print(time.time() - a)

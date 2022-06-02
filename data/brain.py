import scipy.io
import torch
import numpy as np
import os

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Loading brain dataset.

    Args:
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader (torch.evaluate.data.DataLoader): Data loader.
    """

    BrainData.init(root, num_query, num_train)
    query_dataset = BrainData(root, 'query', query_transform())
    train_dataset = BrainData(root, 'train', train_transform())
    retrieval_dataset = BrainData(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class BrainData(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = BrainData.TRAIN_DATA
            self.targets = BrainData.TRAIN_TARGETS
        elif mode == 'query':
            self.data = BrainData.QUERY_DATA
            self.targets = BrainData.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = BrainData.RETRIEVAL_DATA
            self.targets = BrainData.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = torch.from_numpy(self.data[index])
        img = torch.squeeze(img, 2)
        img = torch.unsqueeze(img, 0)
        return img, self.targets[index], index

    def __len__(self):
        return self.data.shape[0]

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    @staticmethod
    def init(root, num_query, num_train):
        """
        Initialize dataset

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        # Load dataset
        data_mat = scipy.io.loadmat("data/data.mat")
        output = np.transpose(data_mat['phenotype'])
        inputs = np.transpose(data_mat['net'])

        output = torch.from_numpy(output).float()
        output = torch.transpose(output, 0, 1)
        inputs = torch.from_numpy(inputs).float()
        inputs = torch.transpose(inputs, 0, 2)
        inputs = torch.unsqueeze(inputs, 3)[:1080]
        output = torch.nn.functional.one_hot(output[:, 2].type(torch.int64), num_classes=2).type(torch.float)
        # output = output[:, 2].type(torch.LongTensor)[:1080]

        # Split dataset
        perm_index = np.random.permutation(inputs.shape[0])
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query:]

        BrainData.QUERY_DATA = inputs[query_index].numpy()
        BrainData.QUERY_TARGETS = output[query_index, :].numpy()

        BrainData.TRAIN_DATA = inputs[train_index].numpy()
        BrainData.TRAIN_TARGETS = output[train_index, :].numpy()

        BrainData.RETRIEVAL_DATA = inputs[retrieval_index].numpy()
        BrainData.RETRIEVAL_TARGETS = output[retrieval_index, :].numpy()

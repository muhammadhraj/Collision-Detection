import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('/Users/muhammadhraj/Downloads/assignment_part1-2/saved/TODO REDUCE SAMPLE training_data.csv', dtype=np.float32, delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
        self.data = self.normalized_data
        self.inputs = torch.from_numpy(self.data[:,0:6])
        self.labels = torch.from_numpy(self.data[:,6])

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        return {'input': self.inputs[idx], 'label': self.labels[idx]}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        self.train_size = int(0.8 * self.nav_dataset.__len__())
        self.test_size = self.nav_dataset.__len__() - self.train_size
        self.train_split, self.test_split = data.random_split(self.nav_dataset, [self.train_size, self.test_size])
        self.train_loader = data.DataLoader(self.train_split, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_split, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()

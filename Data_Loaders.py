import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('/Users/muhammadhraj/Downloads/assignment_part1-2/saved/TODO REDUCE SAMPLE training_data.csv', dtype=np.float32, delimiter=',')

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference
        self.data = self.normalized_data
        self.inputs = torch.from_numpy(self.data[:,0:6])
        self.labels = torch.from_numpy(self.data[:,6])

    def __len__(self):
        return len(self.normalized_data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
            
        return {'input': self.inputs[idx], 'label': self.labels[idx]}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.train_size = int(0.8 * self.nav_dataset.__len__())
        self.test_size = self.nav_dataset.__len__() - self.train_size
        self.train_split, self.test_split = data.random_split(self.nav_dataset, [self.train_size, self.test_size])
        self.train_loader = data.DataLoader(self.train_split, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.test_split, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()

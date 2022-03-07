from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 4000
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = .0005)

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            optimizer.zero_grad()
            loss = loss_function(model.forward(sample['input']), torch.reshape(sample['label'], (len(sample['input']), 1)))
            loss.backward()
            optimizer.step()
        losses.append(loss)
        print(epoch_i)

    torch.save(model.state_dict(), './saved/saved_model.pkl', _use_new_zipfile_serialization=False)
    plt.plot(losses)
    plt.xlabel("no epochs")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    no_epochs =10000
    train_model(no_epochs)


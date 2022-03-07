import torch
import torch.nn as nn
import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_size = 6
        self.hidden_size = 8
        self.output_size = 1
        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.relu(hidden)
        output = self.hidden_to_output(hidden)
        output = self.sigmoid(output)
        return output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        runningLoss = torch.tensor(0.0)
        for idx, sample in enumerate(test_loader):
            runningLoss += loss_function(self.forward(sample['input']), torch.reshape(sample['label'], (len(sample['input']),1)))
        loss = (runningLoss/idx).item()
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_size = 6
        self.hidden1_size = 100
        self.hidden2_size = 50
        self.hidden3_size = 20
        self.output_size = 1
        self.input_to_hidden1 = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden1_to_hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.hidden2_to_hidden3 = nn.Linear(self.hidden2_size, self.hidden3_size)
        self.hidden3_to_output = nn.Linear(self.hidden3_size, self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden1 = self.input_to_hidden1(input)
        hidden1 = self.relu(hidden1)

        hidden2 = self.hidden1_to_hidden2(hidden1)
        hidden2 = self.relu(hidden2)

        hidden3 = self.hidden2_to_hidden3(hidden2)
        hidden3 = self.relu(hidden3)

        output = self.hidden3_to_output(hidden3)
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

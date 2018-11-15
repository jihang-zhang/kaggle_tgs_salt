import torch

sigmoid = torch.nn.Sigmoid()
def accuracy(outputs, targets):
    outputs = sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    result = (outputs == targets).float().sum()
    return result

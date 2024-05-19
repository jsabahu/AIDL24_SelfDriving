import torch

# Function use in train
def binary_accuracy_with_logits(labels, outputs):
    preds = torch.sigmoid(outputs).round()
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc

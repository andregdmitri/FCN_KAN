import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np

class TimeSeriesClassifier(LightningModule):
    def __init__(self, model, optimizer):
        super(TimeSeriesClassifier, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() if model.num_classes > 2 else nn.BCEWithLogitsLoss()
        self.opt = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        logits = self(inputs)

        if self.model.num_classes == 2:
            logits = logits.squeeze(dim=-1).float()
            labels = labels.type(torch.LongTensor)
            y_pred = F.sigmoid(logits).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()

        loss = self.loss_fn(logits, labels.type(torch.LongTensor).to(self.device))

        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        f1 = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
        
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_accuracy', acc, prog_bar=True, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        logits = self(inputs)
        
        if self.model.num_classes == 2:
            logits = logits.squeeze(dim=-1)
            y_pred = F.sigmoid(logits).round()

            y_pred = y_pred.cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
        else:
            y_pred = torch.argmax(logits, dim=1).cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
        loss = self.loss_fn(logits, labels.type(torch.LongTensor).to(self.device))
        
        acc = accuracy_score(y_pred=y_pred, y_true=y_true)
        f1 = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
        
        self.log('loss', loss, prog_bar=False, on_epoch=True)
        self.log('accuracy', acc, prog_bar=False, on_epoch=True)
        self.log("f1", f1, prog_bar=False, on_epoch=True) # type: ignore

        return

    def configure_optimizers(self):
        return self.opt

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, label_mapping=None):
        self.X = X
        self.y = y
        self.label_mapping = label_mapping or self.create_label_mapping(y)
        self.y_mapped = [self.label_mapping[label] for label in y]


    def create_label_mapping(self, y):
        unique_labels = set(y)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        return label_mapping

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.tensor(self.y_mapped[index], dtype=torch.long)

def accuracy(model, dataset):
    train_pred = torch.argmax(model(dataset['train_input']), dim=1)
    test_pred = torch.argmax(model(dataset['test_input']), dim=1)
    
    train_acc = torch.mean((train_pred == dataset['train_label']).float())
    test_acc = torch.mean((test_pred == dataset['test_label']).float())
    
    return train_acc.item(), test_acc.item()

# F1-score function
def f1_score_metric(model, dataset):
    train_pred = torch.argmax(model(dataset['train_input']), dim=1).cpu().numpy()
    test_pred = torch.argmax(model(dataset['test_input']), dim=1).cpu().numpy()
    
    train_f1 = f1_score(dataset['train_label'].cpu().numpy(), train_pred, average='weighted')
    test_f1 = f1_score(dataset['test_label'].cpu().numpy(), test_pred, average='weighted')
    
    return train_f1, test_f1
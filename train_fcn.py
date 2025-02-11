import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from models.fcn import FCNClassifier
from utils import TimeSeriesClassifier, TimeSeriesDataset
from kan import KAN
from aeon.datasets import load_from_ts_file, load_classification
import pandas as pd
from pytorch_lightning.loggers.wandb import WandbLogger

# SETTINGS
NUM_EXPERIMENTS = 5
MAX_EPOCHS = 100

def load_data():
    X_train, y_train = load_from_ts_file("./data/ts_files/train.ts")
    X_test, y_test = load_from_ts_file("./data/ts_files/test.ts")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return X_train, y_train_encoded, X_test, y_test_encoded, len(label_encoder.classes_)


def load_data_from_aeon(dataset_name):
    X_train, y_train = load_classification(dataset_name, split='train')
    X_test, y_test = load_classification(dataset_name, split='test')
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return X_train, y_train_encoded, X_test, y_test_encoded, len(label_encoder.classes_)

def create_dataloaders(X_train, y_train, X_test, y_test):
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    return train_loader, test_loader

def train_and_evaluate(train_loader, test_loader, dimension_num, num_classes, experiment):
    fcn_model = FCNClassifier(dimension_num=dimension_num, activation=nn.ReLU(), num_classes=num_classes)
    optimizer = torch.optim.Adadelta(fcn_model.parameters(), lr=1e-3, eps=1e-8)
    model_classifier = TimeSeriesClassifier(model=fcn_model, optimizer=optimizer)
    wandb_logger = WandbLogger(log_model="all", project="FCN")
    checkpoint_callback = ModelCheckpoint(dirpath="experiments", filename=f"fcn_kan_{experiment}", save_top_k=1, monitor="f1", mode="max")
    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=-1, callbacks=[checkpoint_callback], logger=wandb_logger, enable_model_summary=False)
    trainer.fit(model_classifier, train_loader)
    results = trainer.test(model_classifier, test_loader)
    return results, wandb_logger

def main():
    datasets = [
        "ECG5000", 
        "ECG200", 
        "TwoLeadECG", 
        "ECGFiveDays",
        "NonInvasiveFetalECGThorax1", 
        "NonInvasiveFetalECGThorax2"
    ]
    for dataset in datasets:
        # X_train, y_train, X_test, y_test, num_classes = load_data()
        X_train, y_train, X_test, y_test, num_classes = load_data_from_aeon(dataset)
        dimension_num = X_train.shape[1] if len(X_train.shape) > 1 else 1
        train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)
        results_dict = {'experiment': [], 'accuracy': [], 'f1': []}
        
        for experiment in range(NUM_EXPERIMENTS):
            results, wandb_logger = train_and_evaluate(train_loader, test_loader, dimension_num, num_classes, experiment)
            results_dict['experiment'].append(experiment)
            results_dict['accuracy'].append(results[0]['accuracy'])
            results_dict['f1'].append(results[0]['f1'])
            pd.DataFrame(results_dict).to_csv(f'./results_fcn_{experiment}.csv', index=False)
            wandb_logger.log_metrics({"experiment": experiment})
            wandb_logger.finalize("success")

if __name__ == "__main__":
    main()
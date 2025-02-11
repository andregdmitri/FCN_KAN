import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from aeon.datasets import load_from_ts_file
from sklearn.preprocessing import LabelEncoder
from kan import KAN

# Settings
NUM_EXPERIMENTS = 5
STEPS = 50
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load and preprocess training and testing datasets."""
    train_path, test_path = "./data/ts_files/train.ts", "./data/ts_files/test.ts"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test .ts files not found.")
    
    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    return (torch.from_numpy(X_train.squeeze(1)).float().to(dev),
            torch.from_numpy(y_train_encoded).long().to(dev),
            torch.from_numpy(X_test.squeeze(1)).float().to(dev),
            torch.from_numpy(y_test_encoded).long().to(dev),
            len(np.unique(y_train_encoded)))

def train_kan(train_input, train_label, test_input, test_label, n_timepoints, n_classes):
    """Train the KAN model and return metrics."""
    dataset = {"train_input": train_input, "train_label": train_label, 
               "test_input": test_input, "test_label": test_label}
    
    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())
    
    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())
    
    results_list = []
    for exp in range(NUM_EXPERIMENTS):
        # model = KAN(ckpt_path='./kan_results', width=[[n_timepoints, 0], [50, 0], [30, 0], [n_classes, 0]],
        #             grid=5, k=3, seed=42, device=dev)
        model = KAN(ckpt_path='./kan_results', width=[128, 5, n_classes], grid=5, k=3, seed=42, device=dev)
        results = model.fit(dataset, steps=STEPS, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss(), log=1)
        results_list.append({"experiment": exp, "train_loss": results["train_loss"][-1].item(), 
                             "test_loss": results["test_loss"][-1].item(), "train_acc": results["train_acc"][-1], 
                             "test_acc": results["test_acc"][-1]})
    return results_list

def save_metrics(metrics):
    """Save experiment metrics to a JSON file."""
    with open("kan_metrics.json", "w") as f:
        json.dump(metrics, f)

def main():
    train_input, train_label, test_input, test_label, n_classes = load_data()
    n_timepoints = train_input.shape[1]
    metrics = train_kan(train_input, train_label, test_input, test_label, n_timepoints, n_classes)
    save_metrics(metrics)

if __name__ == "__main__":
    main()
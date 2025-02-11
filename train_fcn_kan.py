import torch
import json
from torch import nn
from torch.utils.data import DataLoader
from models.fcn import GAP1d, FCNClassifier
from kan import KAN
from utils import TimeSeriesClassifier, TimeSeriesDataset
from aeon.datasets import load_from_ts_file
from sklearn.preprocessing import LabelEncoder

# SETTINGS
STEPS = 50
NUM_EXPERIMENTS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings_list, labels_list = [], []
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            x = data
            for layer in model.layers:
                x = layer(x)
                if isinstance(layer, GAP1d):
                    embeddings = x
                    break
            embeddings_list.append(embeddings.cpu())
            labels_list.append(labels.cpu())
    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)


["ECG5000", "ECG200", "TwoLeadECG","ECGFiveDays","NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2"],

def train_kan_model(dataset, embedding_size, n_classes):
    train_losses, test_losses, train_accuracies, test_accuracies, experiment_num = [], [], [], [], []
    for experiment in range(NUM_EXPERIMENTS):
        kan_model = KAN(
            ckpt_path='./kan_fcn',
            width=[[embedding_size, 0], [50, 0], [30, 0], [n_classes, 0]],
            k=3,
            seed=42,
            device=device,
        )
        results = kan_model.fit(
            dataset=dataset,
            steps=STEPS,
            metrics=(lambda: torch.mean((torch.argmax(kan_model(dataset['train_input']), dim=1) == dataset['train_label']).float()),
                     lambda: torch.mean((torch.argmax(kan_model(dataset['test_input']), dim=1) == dataset['test_label']).float())),
            loss_fn=torch.nn.CrossEntropyLoss(),
            log=1,
        )
        experiment_num.append(experiment)
        train_losses.append(results["train_loss"][-1])
        test_losses.append(results["test_loss"][-1])
        train_accuracies.append(results["train_acc"][-1])
        test_accuracies.append(results["test_acc"][-1])
    return experiment_num, train_losses, test_losses, train_accuracies, test_accuracies

def main():
    # Load time series data
    X_train, y_train = load_from_ts_file("./data/ts_files/train.ts")
    X_test, y_test = load_from_ts_file("./data/ts_files/test.ts")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)


    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train_encoded)
    test_dataset = TimeSeriesDataset(X_test, y_test_encoded)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


    # Load pre-trained FCN model
    fcn_model = FCNClassifier(dimension_num=X_train.shape[1], activation=nn.ReLU(), num_classes=num_classes).to(device)
    train_embeddings, train_labels = get_embeddings(fcn_model, train_loader, device)
    test_embeddings, test_labels = get_embeddings(fcn_model, test_loader, device)
    dataset = {
        "train_input": train_embeddings.to(device),
        "train_label": train_labels.to(device),
        "test_input": test_embeddings.to(device),
        "test_label": test_labels.to(device),
    }
    experiment_num, train_losses, test_losses, train_accuracies, test_accuracies = train_kan_model(
        dataset, train_embeddings.shape[1], len(torch.unique(train_labels))
    )
    metrics = {
        'experiment': experiment_num,
        "train_loss": [loss.item() for loss in train_losses],
        "test_loss": [loss.item() for loss in test_losses],
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
    }
    with open("kan_fcn_metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()

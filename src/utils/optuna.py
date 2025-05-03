from utils.yaml import Config
from utils.dataset import transforms
from utils.dataset import SportsDataset
from torch.utils.data import DataLoader
import optuna
import torch
import torch.nn as nn
from tqdm import tqdm
from architectures.custom import CustomCNN  # Import your model architecture
from functools import partial
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config.load()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_csv_path = cfg['train_csv_path']
test_csv_path = cfg['test_csv_path']
dataset_path = cfg['dataset_path']
NUM_CLASSES = cfg['NUM_CLASSES']
EPOCHS = cfg['EPOCHS']
BETAS = cfg['BETAS']


def objective(trial, model_cls=CustomCNN):
    # Define the hyperparameters to search
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical(
        'optimizer', ['Adam', 'SGD', 'RMSprop'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Setup model
    model = model_cls()  # your model here
    model = model.to(device)

    # Dataloader
    train_dataset = SportsDataset(
        csv_file=train_csv_path, file_path=dataset_path, split='train', transform=transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop (1-3 epochs is enough for Optuna)
    for epoch in range(3):
        model.train()
        running_loss = 0.0

        # Training progress bar
        train_pbar = tqdm(
            train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}/3 (Train)")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'loss': running_loss/(train_pbar.n+1)})

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0

    # Validation progress bar
    with torch.no_grad():
        val_pbar = tqdm(train_loader, desc=f"Trial {trial.number}, Validation")
        for inputs, targets in val_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Update progress bar with current accuracy
            val_pbar.set_postfix({'acc': correct/total})

    accuracy = correct / total
    trial.set_user_attr('final_accuracy', accuracy)
    return accuracy


def run_optuna_study(objective, n_trials=30, direction='maximize', model_to_search=CustomCNN):

    study = optuna.create_study(direction=direction)
    study.optimize(partial(objective, model_cls=model_to_search),
                   n_trials=n_trials)

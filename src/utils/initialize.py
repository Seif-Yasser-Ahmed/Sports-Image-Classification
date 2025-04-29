from src.architectures.custom import CustomCNN
from dataset import SportsDataset
from src.utils.early_stopping import EarlyStopping
from src.utils.yaml import Config
from src.utils.weights import init_weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch
import torch.nn as nn
import os
from torchviz import make_dot
from src.utils.dataset import transforms

cfg = Config.load()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_csv_path = cfg['train_csv_path']
test_csv_path = cfg['test_csv_path']
dataset_path = cfg['dataset_path']
NUM_CLASSES = cfg['NUM_CLASSES']
EPOCHS = cfg['EPOCHS']
BETAS = cfg['BETAS']


def initialize_model(model_class=CustomCNN, num_classes=NUM_CLASSES, input_channels=3, input_size=(224, 224), dropout=0.5, hidden_dim=64, pretrained_weights=None, weight_init_method='custom'):
    # classifier=CustomCNN(num_classes=NUM_CLASSES, input_channels=3, input_size=(224, 224),dropout=0.5,hidden_dim=64)
    model = model_class(num_classes=num_classes, input_channels=input_channels,
                        input_size=input_size, dropout=dropout, hidden_dim=hidden_dim)
    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))
    else:
        init_weights(model, init_type=weight_init_method)
    model.to(device)
    return model


def initialize_optimizer(model, optimizer, learning_rate=0.001, betas=(0.9, 0.999), weight_decay=0.0001):
    if optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, betas=betas)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(
        ), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=weight_decay, centered=False)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0,
                                  weight_decay=weight_decay, initial_accumulator_value=0, eps=1e-10)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    return optimizer


def initialize_loss_function(loss_fn='crossentropy'):
    if loss_fn == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_fn == 'KLDivLoss':
        criterion = nn.KLDivLoss(reduction='batchmean')
    elif loss_fn == 'svm':
        criterion = nn.MultiMarginLoss(margin=1.0)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    criterion.to(device)
    return criterion


def initialize_lr_scheduler(optimizer, scheduler_type='step', step_size=10, gamma=0.1):
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=0)
    elif scheduler_type == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=EPOCHS)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return scheduler


def initialize_data_loaders(train_csv_path, test_csv_path, dataset_path, transforms):
    train_dataset = SportsDataset(
        csv_file=train_csv_path, file_path=dataset_path, split='train', transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = SportsDataset(
        csv_file=test_csv_path, file_path=dataset_path, split='test', transform=transforms)

    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_loader, val_loader


def initialize_tensorboard_writer(log_directory, run_name):
    # Create a directory for the logs if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(log_directory, run_name))
    return writer


def initialize_early_stopping(patience=5, delta=0):
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    return early_stopping


def draw_save_graph(model, name='custom_arch_graph'):
    # Create the directory if it doesn't exist
    graph_dir = '../models/graphs'
    os.makedirs(graph_dir, exist_ok=True)

    # Create full path for the graph
    path = os.path.join(graph_dir, name)

    # Create a graph of the model
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    dot = make_dot(model(sample_input), params=dict(
        list(model.named_parameters())))

    # Save the graph as a PNG file
    dot.format = 'png'
    dot.render(filename=path, cleanup=True)
    print(f"Graph saved to {path}.png")


def initialize_all(run_name='custom_arch_graph', model_class=CustomCNN, num_classes=NUM_CLASSES, input_channels=3, input_size=(224, 224), dropout=0.5, hidden_dim=64, betas=BETAS, optimizer='adam', learning_rate=0.001, weight_decay=0.0001, loss_fn='crossentropy', scheduler_type=None, log_directory='../logs'):
    train_loader, val_loader = initialize_data_loaders(
        train_csv_path, test_csv_path, dataset_path, transforms)
    criterion = initialize_loss_function(loss_fn=loss_fn)
    model = initialize_model(model_class, num_classes,
                             input_channels, input_size, dropout, hidden_dim)
    draw_save_graph(model, name=run_name)
    optimizer = initialize_optimizer(
        model, optimizer, learning_rate=learning_rate, betas=betas, weight_decay=weight_decay)
    if scheduler_type:
        scheduler = initialize_lr_scheduler(
            optimizer, scheduler_type=scheduler_type, step_size=25, gamma=0.1)
    else:
        scheduler = None
    writer = initialize_tensorboard_writer(log_directory, run_name)
    early_stopping = initialize_early_stopping()

    return model, optimizer, criterion, scheduler, writer, early_stopping, train_loader, val_loader
    # initialize_checkpoint_manager()
    # initialize_logger()

import ray
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict
from filelock import FileLock
from ray import tune
from ray.tune import CLIReporter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- HET MODEL ---
class ConfigurableCNN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        layers = []
        in_channels = 1 # FashionMNIST is zwart-wit
        num_filters = config.get("num_filters", 32)
        
        # We bouwen de lagen dynamisch op
        for _ in range(config.get("num_conv_layers", 2)):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            
            # --- HYPOTHESE TEST ---
            # We testen of Batchnorm helpt bij hoge learning rates
            if config.get("use_batchnorm", False):
                layers.append(nn.BatchNorm2d(num_filters))
                
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            
            in_channels = num_filters
            num_filters *= 2
            
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.get("dropout", 0.2)),
            nn.Linear(in_channels, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- DATA LADEN ---
def load_data(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # FileLock voorkomt dat meerdere ray-workers tegelijk downloaden
    with FileLock(Path(data_dir) / ".lock"):
        train_data = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        
    # We gebruiken een kleinere subset (10k train, 5k valid) voor snelheid
    total_len = len(train_data)
    train_len = 10000
    val_len = 5000
    rest = total_len - train_len - val_len
    
    train_subset, val_subset, _ = random_split(train_data, [train_len, val_len, rest])
    return train_subset, val_subset

# --- TRAINING LOOP ---
def train_func(config: Dict):
    # 1. Setup Data
    data_dir = config.get("data_dir")
    train_subset, val_subset = load_data(data_dir)
    
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=int(config["batch_size"]), shuffle=False)

    # 2. Setup Model
    model = ConfigurableCNN(config) 
    
    # Check voor GPU/MPS (Apple Silicon)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    model.to(device)

    # 3. Setup Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # 4. Trainen (10 epochs)
    for epoch in range(10): 
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 5. Valideren
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        
        # 6. Rapporteren aan Ray (FIX: Gebruik een dictionary!)
        tune.report({"accuracy": accuracy, "training_iteration": epoch + 1})

if __name__ == "__main__":
    ray.init()

    # Locaties instellen
    base_dir = Path(".").resolve()
    data_dir = base_dir / "data"
    tune_dir = base_dir / "logs" / "ray"
    
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    # Configuraties voor het experiment
    config = {
        "data_dir": str(data_dir),
        "batch_size": 64,
        "num_conv_layers": 2,
        "num_filters": 32,
        "dropout": 0.2,
        
        # --- HYPOTHESE VARIABELEN ---
        # Grid search: test verplicht True én False
        "use_batchnorm": tune.grid_search([True, False]),
        
        # Loguniform: test learning rates van heel klein tot heel groot
        "lr": tune.loguniform(1e-4, 1e-1),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("accuracy")

    print("Start training... dit duurt een paar minuten...")

    analysis = tune.run(
        train_func,
        config=config,
        metric="accuracy",
        mode="max",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=20, # Totaal 20 runs
        verbose=1,
    )

    print("Klaar! Beste config:", analysis.best_config)
    ray.shutdown()
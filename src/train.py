import torch
from pathlib import Path
from dataset import CapuchinBirdCallDataset
from torch.utils.data import DataLoader
from models import EfficientNetB0Classifier
import engine
import utils

# Setup device-agnostic code
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = "mps"  # Apple GPU
else:
    device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

WORKERS = 0

DATA_FOLDER = Path("./data")
CAPUCHIN_CLIPS_FOLDER = DATA_FOLDER / "Parsed_Capuchinbird_Clips"
NOT_CAPUCHIN_CLIPS_FOLDER = DATA_FOLDER / "Parsed_Not_Capuchinbird_Clips"
if not CAPUCHIN_CLIPS_FOLDER.exists() and not NOT_CAPUCHIN_CLIPS_FOLDER.exists():
    raise Exception("Data folder not does not exist")

SAMPLE_RATE = 16000
NUM_FRAMES = 48000

capuchin_call_dataset = CapuchinBirdCallDataset(
    CAPUCHIN_CLIPS_FOLDER, NOT_CAPUCHIN_CLIPS_FOLDER, SAMPLE_RATE, NUM_FRAMES
)

train_test_split_generator = torch.Generator().manual_seed(42)
train_dataset, test_dataset = torch.utils.data.random_split(
    capuchin_call_dataset, [0.8, 0.2], generator=train_test_split_generator
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=32, num_workers=WORKERS, shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=32, num_workers=WORKERS, shuffle=True
)

model = EfficientNetB0Classifier().to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

writer = utils.create_writer("Baseline", "EfficientNetB0Classifier")

engine.train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    epochs=18,
    device=device,
    writer=writer,
)

utils.save_model(
    model, target_dir="./artifacts", model_name="EfficientNetB0Classifier.pth"
)

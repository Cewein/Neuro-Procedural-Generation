#!/usr/bin/env python3

# %%--------------------------
# Import Libraries
# ---------------------------
# This script generates a 2D tile map using a trained RNN model.
# It includes functions for loading data, training the model, and generating tile maps.
# The generated tile maps are displayed as mosaics using PIL and matplotlib.
# The model is saved to a specified path after training.
# ---------------------------

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# %%--------------------------
# Global Settings and Paths
# ---------------------------
TILE_ROOT      = "output_tiles"                             # root where all unique_tile_*.png live
TILE_FOLDER    = TILE_ROOT                                  # used by display_tile_map
MAP_DAT_DIR    = os.path.join(TILE_ROOT, "bin")             # directory containing all .dat files
MODEL_SAVE_PATH = "model_rnn_2D_512_512_1.pth"
MAP_SHAPE      = (32, 32)
TILE_SIZE      = 16

# %%--------------------------
# Utility Functions
# ---------------------------

def load_tile_map(filepath, map_shape=MAP_SHAPE, dtype=np.uint32):
    """
    Load and reshape tile map data from a binary file.
    """
    data = np.fromfile(filepath, dtype=dtype)
    if data.size != np.prod(map_shape):
        raise ValueError(f"Data size in {filepath} does not match expected shape {map_shape}.")
    return data.reshape(map_shape)


def display_tile_map(tile_map_2d, title="Generated Tile Map"):
    """
    Create and display a mosaic image for a given 2D tile map.
    """
    sample_tile_path = os.path.join(TILE_FOLDER, "unique_tile_0.png")
    try:
        sample_tile = Image.open(sample_tile_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Sample tile not found: {sample_tile_path}")
    tile_width, tile_height = sample_tile.size

    num_rows, num_cols = tile_map_2d.shape
    mosaic = Image.new("RGB", (num_cols * tile_width, num_rows * tile_height))

    for row in range(num_rows):
        for col in range(num_cols):
            tile_id = tile_map_2d[row, col]
            tile_path = os.path.join(TILE_FOLDER, f"unique_tile_{tile_id}.png")
            try:
                tile_img = Image.open(tile_path)
            except FileNotFoundError:
                # Use a red placeholder if the tile image is missing.
                tile_img = Image.new("RGB", (tile_width, tile_height), color=(255, 0, 0))
            mosaic.paste(tile_img, (col * tile_width, row * tile_height))

    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(mosaic))
    plt.axis("off")
    plt.title(title)
    plt.show()

# %%--------------------------
# Neighbor Kernel & Sampling
# ---------------------------
# Define neighbor offsets (col, row)
NEIGHBOR_KERNEL = [(0, -1), (1, -1), (2, -1), (-1, 0)]
SEQUENCE_LENGTH = len(NEIGHBOR_KERNEL) + 1  # neighbors + target

# This token should be outside the range of tile IDs in the dataset.
PAD_TOKEN = 594

def sample_tile(tile_map, col, row):
    """
    Safely sample a tile from the tile map using column and row indices.
    Returns PAD_TOKEN if indices are out of range.
    """
    num_rows, num_cols = tile_map.shape
    if col < 0 or col >= num_cols or row < 0 or row >= num_rows:
        return PAD_TOKEN
    return tile_map[row, col]


def extract_training_data(tile_map, neighbor_kernel=NEIGHBOR_KERNEL):
    """
    Generate training sequences by sliding over the tile map.
    For each (row, col), create a sequence of neighbor samples followed by the target tile.
    Returns a 1D numpy array of length (num_tiles * sequence_length).
    """
    num_rows, num_cols = tile_map.shape
    sequences = []
    for row in range(num_rows):
        for col in range(num_cols):
            seq = []
            for dx, dy in neighbor_kernel:
                seq.append(sample_tile(tile_map, col + dx, row + dy))
            # Append the current tile as the final token in the sequence.
            seq.append(sample_tile(tile_map, col, row))
            sequences.append(seq)
    return np.array(sequences).flatten()

# %%--------------------------
# Data Preparation
# ---------------------------
# Scan MAP_DAT_DIR for all .dat files

def extract_map_size_from_path(filepath):
    """
    Given a filepath whose basename contains WIDTH-HEIGHT (e.g. '..._1024-1024_map.dat'),
    return a tuple of two ints: (width, height).
    """
    # Extract the filename
    filename = os.path.basename(filepath)
    # Use a regex to find two numbers separated by a dash
    match = re.search(r'_(\d+)-(\d+)_', filename)
    if not match:
        raise ValueError(f"Could not find map dimensions in '{filename}'")
    width, height = map(int, match.groups())
    return width//TILE_SIZE, height//TILE_SIZE

map_paths = sorted(
    os.path.join(MAP_DAT_DIR, fname)
    for fname in os.listdir(MAP_DAT_DIR)
    if fname.lower().endswith(".dat")
)

if not map_paths:
    raise RuntimeError(f"No .dat files found in {MAP_DAT_DIR}")

all_sequences = []
print(f"Found {len(map_paths)} .dat file(s) for training:")
for path in map_paths:
    print(f"  • {path}")
    tile_map = load_tile_map(path, map_shape=extract_map_size_from_path(path))
    seq_flat = extract_training_data(tile_map)
    all_sequences.append(seq_flat)

# Concatenate into one flat training array
training_data_flat = np.concatenate(all_sequences)
print(f"Total training tokens from all maps: {training_data_flat.size}")

# Build vocabulary and PAD token
vocabulary = np.unique(training_data_flat)
vocab_size = len(vocabulary)
print(f"Vocabulary size (including PAD): {vocab_size}")

# %%--------------------------
# Dataset and DataLoader
# ---------------------------

BATCH_SIZE = 256

class TileMapDataset(Dataset):
    def __init__(self, data, sequence_length, positional_info):
        self.data = data
        self.sequence_length = sequence_length
        self.positional_info = positional_info

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, index):
        start = index * self.sequence_length
        inputs = self.data[start : start + self.sequence_length - 1]
        targets = self.data[start + 1 : start + self.sequence_length]
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            self.positional_info
        )

# Precompute positional indices
positional_info = torch.arange(SEQUENCE_LENGTH - 1, dtype=torch.long)
dataset = TileMapDataset(training_data_flat, SEQUENCE_LENGTH, positional_info)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# %%--------------------------
# Model Definition
# ---------------------------

class TileModelRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512, n_layers=1, dropout=0.0):
        super(TileModelRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use positional embeddings based on sequence positions (0 to sequence_length-2)
        self.positional_embedding = nn.Embedding(SEQUENCE_LENGTH - 1, embedding_dim)
        # Bidirectional GRU layer
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x, positions):
        token_embedded = self.embedding(x)

        # Only add a batch dimension if positions is 1D
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(x.size(0), -1)
        pos_embedded = self.positional_embedding(positions)

        embedded = token_embedded + pos_embedded
        rnn_output, _ = self.rnn(embedded)
        return self.fc(rnn_output)

# Set device using abstraction for CUDA/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TileModelRNN(vocab_size).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# %%--------------------------
# Training Loop
# ---------------------------

learning_rate = 3e-4
num_epochs = 6000

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()

loss_visualization = []

pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    epoch_loss = 0.0
    for inputs, targets, positions in dataloader:
        inputs, targets, positions = inputs.to(device), targets.to(device), positions.to(device)
        outputs = model(inputs, positions)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    pbar.set_postfix(avg_loss=f"{avg_loss}")
    loss_visualization.append(avg_loss)

# Save model state instead of the full model.
torch.save(model.state_dict(), MODEL_SAVE_PATH)

plt.semilogy(loss_visualization)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("training_loss.png")
plt.show()

# %%--------------------------
# Generation Functions
# ---------------------------

def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)

def apply_repetition_penalty(logits, generated_map, col, row, penalty_value, map_length):
    """
    Apply repetition penalty to logits based on recently generated tiles.
    This version penalizes tokens from previously generated neighbors
    in the current row and column.
    """
    def get_tile(col_idx, row_idx):
        if col_idx < 0 or col_idx >= map_length or row_idx < 0 or row_idx >= map_length:
            return None
        return generated_map[row_idx * map_length + col_idx]
    
    # Penalize tokens in the current row (look backwards up to 8 tiles)
    for offset in range(1, min(9, col + 1)):
        token = get_tile(col - offset, row)
        if token is not None:
            logits[token] *= penalty_value
    # Penalize tokens in the current column (look backwards up to 8 tiles)
    for offset in range(1, min(9, row + 1)):
        token = get_tile(col, row - offset)
        if token is not None:
            logits[token] *= penalty_value
    return logits

def top_k_sampling(probs, k):
    top_probs, top_indices = torch.topk(probs, k)
    top_probs = top_probs / top_probs.sum()
    return top_indices[torch.multinomial(top_probs, 1).item()]

def generate_tilemap(model, size, temperature=0.8, rep_penalty=0.9, top_k=100):
    """
    Generate a tile map of given size using the trained model.
    Generation is done in an auto-regressive, tile-by-tile manner.
    """
    model.eval()
    generated = [PAD_TOKEN] * (size * size)
    
    def get_generated_tile(col, row):
        if col < 0 or col >= size or row < 0 or row >= size:
            return PAD_TOKEN
        return generated[row * size + col]
    
    with torch.no_grad():
        # Iterate in row-major order to ensure dependencies are generated
        for row in range(size):
            for col in range(size):
                seq = [get_generated_tile(col + dx, row + dy) for dx, dy in NEIGHBOR_KERNEL]
                input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                pos_info = torch.arange(len(seq), dtype=torch.long).unsqueeze(0).to(device)

                output = model(input_seq, pos_info)
                logits = output[0, -1]
                logits = apply_repetition_penalty(logits, generated, col, row, rep_penalty, size)
                probs = softmax_with_temperature(logits, temperature)
                next_tile = top_k_sampling(probs, top_k)
                generated[row * size + col] = next_tile.item() if isinstance(next_tile, torch.Tensor) else next_tile
                
    return generated

# %%--------------------------
# Generate and Display Tile Map
# ---------------------------
SIZE = 64
TEMPERATURE = 0.6
REPETITION_PENALTY = 0.95
TOP_K = 10

tile_map_generated = generate_tilemap(model, SIZE,
                                      temperature=TEMPERATURE,
                                      rep_penalty=REPETITION_PENALTY,
                                      top_k=TOP_K)
tile_map_2d = np.array(tile_map_generated).reshape((SIZE, SIZE))
print(f"Generated tile map of size {SIZE}×{SIZE}")

display_tile_map(tile_map_2d, title="Generated Tile Map")

# %%

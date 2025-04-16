
# %%--------------------------
# Import Libraries
# ---------------------------
# This script generates a 2D tile map using a trained RNN model.
# It includes functions for loading data, training the model, and generating tile maps.
# The generated tile maps are displayed as mosaics using PIL and matplotlib.
# The model is saved to a specified path after training.
# ---------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

# %%--------------------------
# Global Settings and Paths
# ---------------------------
TILE_FOLDER = "unique_tiles"
MAP_DAT_PATH = "map.dat"
MODEL_SAVE_PATH = "model_rnn_2D_2.pth"
MAP_SHAPE = (32, 32)

# %%--------------------------
# Utility Functions
# ---------------------------

def load_tile_map(filepath, map_shape=MAP_SHAPE, dtype=np.uint32):
    """
    Load and reshape tile map data from a binary file.
    """
    data = np.fromfile(filepath, dtype=dtype)
    if data.size != np.prod(map_shape):
        raise ValueError("Data size does not match expected shape.")
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
# Data Preparation
# ---------------------------

# Load the tile map
tile_map_data = load_tile_map(MAP_DAT_PATH)
print(f"Tile map shape: {tile_map_data.shape}")

# Compute vocabulary information
vocabulary = np.unique(tile_map_data)
vocab_size = len(vocabulary)
PAD_TOKEN = vocab_size  # Padding token at the end of vocabulary index range
vocab_size += 1
print(f"Vocabulary Size (including PAD): {vocab_size}")

# Define neighbor kernel (using (dx, dy) offsets; here we use (col, row))
#NEIGHBOR_KERNEL = [(0, -2), (1, -2), (2, -2),(0, -1), (1, -1), (2, -1), (-1, 0)]
NEIGHBOR_KERNEL = [(-2, -2), (-1, -2), (0, -2),(1, -2), (2, -2), 
                   (-2, -1), (-1, -1), (0, -1),(1, -1), (2, -1),
                   (-2, 0), (-1, 0)]
# This kernel samples 24 previous neighbors in a 5x5 grid centered on the target tile.
SEQUENCE_LENGTH = len(NEIGHBOR_KERNEL) + 1  # Neighbor tokens plus the target tile

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

training_data_flat = extract_training_data(tile_map_data)
print(f"Total training tokens: {len(training_data_flat)}")

# %%--------------------------
# Dataset and DataLoader
# ---------------------------

class TileMapDataset(Dataset):
    def __init__(self, data, sequence_length, positional_info):
        """
        data: Flattened 1D numpy array of training tokens.
        sequence_length: Number of tokens per sequence.
        positional_info: Precomputed position indices tensor of shape (sequence_length - 1,)
        """
        self.data = data
        self.sequence_length = sequence_length
        self.positional_info = positional_info  # This is constant for all samples
        
    def __len__(self):
        return len(self.data) // self.sequence_length
    
    def __getitem__(self, index):
        start = index * self.sequence_length
        # Input is the first (sequence_length - 1) tokens; target is the shifted sequence.
        inputs = self.data[start : start + self.sequence_length - 1]
        targets = self.data[start + 1 : start + self.sequence_length]
        return (torch.tensor(inputs, dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
                self.positional_info)

# Precompute positional indices: here we use a simple sequential index for each token in the input sequence.
positional_info = torch.arange(SEQUENCE_LENGTH - 1, dtype=torch.long)
dataset = TileMapDataset(training_data_flat, SEQUENCE_LENGTH, positional_info)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

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
        """
        x: Tensor of token indices, shape (batch, seq_len)
        positions: Tensor of position indices. It can be either:
                   - A 1D tensor of shape (seq_len,) that will be expanded to (batch, seq_len), or 
                   - Already batched of shape (batch, seq_len)
        """
        token_embedded = self.embedding(x)

        # Only add a batch dimension if positions is 1D
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(x.size(0), -1)
        pos_embedded = self.positional_embedding(positions)

        embedded = token_embedded + pos_embedded
        rnn_output, _ = self.rnn(embedded)
        output = self.fc(rnn_output)
        return output


# Set device using abstraction for CUDA/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TileModelRNN(vocab_size).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_params}')

# %%--------------------------
# Training Loop
# ---------------------------

learning_rate = 3e-4
num_epochs = 5000

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

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
    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

# Save model state instead of the full model.
torch.save(model.state_dict(), MODEL_SAVE_PATH)

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
    sampled_idx = torch.multinomial(top_probs, 1).item()
    return top_indices[sampled_idx]

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
                # Build the input sequence from the neighbor kernel
                seq = []
                for dx, dy in NEIGHBOR_KERNEL:
                    seq.append(get_generated_tile(col + dx, row + dy))
                input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
                pos_info = torch.arange(len(seq), dtype=torch.long).to(device).unsqueeze(0)
                
                output = model(input_seq, pos_info)
                logits = output[0, -1]  # Take logits from the final time step
                
                logits = apply_repetition_penalty(logits, generated, col, row, rep_penalty, size)
                probs = softmax_with_temperature(logits, temperature)
                next_tile = top_k_sampling(probs, top_k)
                generated[row * size + col] = next_tile.item() if isinstance(next_tile, torch.Tensor) else next_tile
    return generated

# %%--------------------------
# Generate and Display Tile Map
# ---------------------------

SIZE = 32
TEMPERATURE = 0.5
REPETITION_PENALTY = 0.95
TOP_K = 10

tile_map_generated = generate_tilemap(model, SIZE, temperature=TEMPERATURE,
                                      rep_penalty=REPETITION_PENALTY, top_k=TOP_K)
tile_map_2d = np.array(tile_map_generated).reshape((SIZE, SIZE))
print(f"Generated tile map: {tile_map_generated}")

display_tile_map(tile_map_2d, title="Generated Tile Map")

# %%

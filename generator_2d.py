# %%

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

#%% 

def display_tile_map(tile_map_2d):
    tile_folder = "unique_tiles"

# Determine tile dimensions by loading a sample tile.
    sample_tile_path = os.path.join(tile_folder, "unique_tile_0.png")
    sample_tile = Image.open(sample_tile_path)
    tile_width, tile_height = sample_tile.size

# Get dimensions of the tile map grid.
    num_rows, num_cols = tile_map_2d.shape

# Create a blank image for the mosaic.
    mosaic_width = num_cols * tile_width
    mosaic_height = num_rows * tile_height
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

# Iterate over each tile position in the tile map.
    for row in range(num_rows):
        for col in range(num_cols):
            tile_id = tile_map_2d[row, col]
            tile_path = os.path.join(tile_folder, f"unique_tile_{tile_id}.png")
            try:
                tile_img = Image.open(tile_path)
            except FileNotFoundError:
            # If the tile image file is missing, create a placeholder image.
                tile_img = Image.new("RGB", (tile_width, tile_height), color=(255, 0, 0))
        # Paste the tile image at the appropriate location.
            mosaic.paste(tile_img, (col * tile_width, row * tile_height))

# Display the mosaic using matplotlib.
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(mosaic))
    plt.axis("off")
    plt.title("Generated Tile Map")
    plt.show()

#%%

tile_map_data = np.fromfile("map.dat", dtype=np.uint32)

tile_map_data = tile_map_data.reshape(32,32)

tile_map_size = tile_map_data.shape

vocabulary = np.unique(tile_map_data)
vocabulary_size = np.sum(np.ones_like(vocabulary))

#%%
print(f"map size: {tile_map_size}")
print(f"Size of vocabulary (unique tiles): {np.sum(np.ones_like(np.unique(tile_map_data)))}")
display_tile_map(tile_map_data)

# %%
PAD_TOKEN = vocabulary_size
vocabulary_size += 1  # Increase vocabulary size to account for padding token

def sample_tile(tile_map_data, x, y, sequence_length=4):
    height, width = tile_map_data.shape
    
    # Check if we can sample 4 tiles horizontally from (x, y)
    if x >= width or y >= height:
        return PAD_TOKEN
    if x < 0 or y < 0:
        return PAD_TOKEN
    
    # Extract tile
    tile = tile_map_data[y, x]
    return tile


# %%
neighbor_kernel = [(0, -1), (1, -1), (2, -1), (-1, 0)]
sequence_length = len(neighbor_kernel) + 1

# Iterate over the tile map data to sample sequences of tiles
data = []
sample_count = 0

# Switch the loops to row-major order (y: row, x: column)
for x in range(tile_map_size[0]):      # tile_map_size[0] is the number of rows
    for y in range(tile_map_size[1]):  # tile_map_size[1] is the number of columns
        values = []
        # Apply the neighbor kernel in the same order as used in generation
        for dx, dy in neighbor_kernel:
            # Here, (x+dx, y+dy): x is the column and y is the row,
            # matching sample_tile() which returns tile_map_data[y, x]
            tile = sample_tile(tile_map_data, x + dx, y + dy, sequence_length)
            values.append(tile)
        
        # Print for debugging (optional)
        print(f"Sampled sequence at ({x}, {y}): {values}")
        sample_count += 1
        data.extend(values)

data = np.array(data)

print(f"Total samples: {sample_count}")
print(f"Data shape: {data.shape}")
print(f"Data: {data}")

# %%

# print(data[6])
# print(data[6,:3,:3])
# print(data[6,1:,1:])

# print(np.array(data[6,:3,:3]).flatten())

# %%
batch_size = 256
class TileMapDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        # Calculate the number of sequences in the dataset
        # we divide by sequence_length since in getitem we return n sequences of length sequence_length
        # n is sequence_length also
        v = len(self.data) // self.sequence_length
        return v

    def __getitem__(self, index):
        return (torch.tensor(self.data[index * self.sequence_length : index * self.sequence_length + self.sequence_length - 1], dtype=torch.long),
                torch.tensor(self.data[index * self.sequence_length + 1 : index * self.sequence_length + self.sequence_length], dtype=torch.long),
                torch.tensor(neighbor_kernel, dtype=torch.long))
dataset = TileMapDataset(data, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%

embedding_dim = 512
hidden_dim = 512
n_layers = 1

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        
        self.positional_embeddings = nn.Embedding(sequence_length - 1, embedding_dim)
        self.position_to_idx = {}
        for i, pos in enumerate(neighbor_kernel):
            self.position_to_idx[pos] = i
    
    def forward(self, sequence_length):
        indices = [self.position_to_idx[pos] for pos in neighbor_kernel]
        
        indices = torch.tensor(indices, dtype=torch.long).to(sequence_length.device).unsqueeze(0).repeat(sequence_length.size(0), 1)
        
        return self.positional_embeddings(indices)

class TileModelRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.0):
        super(TileModelRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x, positions):
        token_embedded = self.embedding(x)
        pos_embedded = self.positional_embedding(positions)
        
        embedded = token_embedded + pos_embedded
        rnn_output, _ = self.rnn(embedded)
        
        output = self.fc(rnn_output)
        return output
    
model = TileModelRNN(vocabulary_size, embedding_dim, hidden_dim, vocabulary_size, n_layers).cuda()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}') # Number of parameters: 152405
# %%

learning_rate = 3e-4
num_epochs = 5000

criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # Ignore padding token during loss calculation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    epoch_loss = 0.0
    for i, (inputs, targets, positions) in enumerate(dataloader):
        inputs, targets, positions = inputs.cuda(), targets.cuda(), positions.cuda()
        outputs = model(inputs, positions)
        loss = criterion(outputs.view(-1, vocabulary_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    # Update the progress bar's postfix with the average loss at the end of the epoch.
    pbar.set_postfix(avg_loss=f"{avg_loss}")


torch.save(model, f'model_rnn_2D_2.pth')
# %%


def softmax_with_temperature(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)

def apply_repetition_penalty(logits, data, x, y, penalty_value, map_size=64):
    def sample_map(x, y):
        if x < 0 or x >= map_size or y < 0 or y >= map_size:
            return map_size - 1
        return data[y * map_size + x]

    for dx in range(-8, 0):
        logits[sample_map(x + dx, y)] *= penalty_value
    for dy in range(-8, 0):
        logits[sample_map(x, y + dy)] *= penalty_value

    return logits

def top_k_probs(probs, k):
    top_k_values, top_k_indices = torch.topk(probs, k)
    top_k_probs = top_k_values / top_k_values.sum()
    return top_k_probs, top_k_indices

def sample_from_top_k(probs, k):
    top_probs, top_indices = top_k_probs(probs, k)
    sampled_index = torch.multinomial(top_probs, 1).item()
    next_token = top_indices[sampled_index]
    return next_token

def generate_tilemap(model, ix, iy, length=16, temperature=0.8, rep_penalty=0.9):
    generated_map = [PAD_TOKEN] * (length * length)

    def sample_map(x, y):
        if x < 0 or x >= length or y < 0 or y >= length:
            return PAD_TOKEN
        return generated_map[y * length + x]

    with torch.no_grad():
        for y in range(length):
            for x in range(length):
                sequence = []
                for dx, dy in neighbor_kernel:
                    sequence.append(sample_map(x+dx, y+dy))

                input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).cuda()
                device = input_tensor.device
                raw_output = model(input_tensor,torch.tensor(neighbor_kernel, dtype=torch.long, device=device))
                raw_output = raw_output[0, -1]

                raw_output = apply_repetition_penalty(raw_output, generated_map, x, y, penalty_value=rep_penalty, map_size=length)
                combined_probs = softmax_with_temperature(raw_output, temperature)
                
                top_5_likely = torch.topk(combined_probs, vocabulary_size - 1)
                #next_token = torch.multinomial(combined_probs, 1).item()
                next_token = sample_from_top_k(combined_probs, K_TOP)

                generated_map[y * length + x] = next_token.item()
    
    return generated_map

SIZE = 32
TEMPERATURE = 0.4
REPETITION_PENALTY = 0.95
K_TOP = 100                

model.eval()
tile_map_gen = generate_tilemap(model, 0, 0,SIZE)



print(f"Generated tile map: {tile_map_gen}")

tile_map_2d = np.array(tile_map_gen).reshape((SIZE, SIZE))


display_tile_map(tile_map_2d)


# %%

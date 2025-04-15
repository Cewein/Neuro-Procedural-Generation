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

tile_map_data = np.fromfile("map.dat", dtype=np.uint32)

tile_map_data = tile_map_data.reshape(32,32)

tile_map_size = tile_map_data.shape

vocabulary = np.unique(tile_map_data)
vocabulary_size = np.sum(np.ones_like(vocabulary))

#%%
print(f"map size: {tile_map_size}")
print(f"Size of vocabulary (unique tiles): {np.sum(np.ones_like(np.unique(tile_map_data)))}")

# %%

def sample_tiles(tile_map_data, x, y, sequence_length=4):
    height, width = tile_map_data.shape
    
    # Check if we can sample 4 tiles horizontally from (x, y)
    if x + sequence_length <= width:
        # Extract the 4-tile sequence
        sequence = tile_map_data[y, x:x+sequence_length]
        return sequence
    
    # If we can't sample 4 tiles (near the edge), return None
    return None

# %%

sequence_length = 4
sample_count = 0
data = []

# Iterate over the tile map data to sample sequences of tiles
for x in range((tile_map_size[0] - sequence_length)):
    for y in range(tile_map_size[1]):

        # Sample a sequence of tiles starting from (x, y)
        sequence = sample_tiles(tile_map_data, x, y, sequence_length)

        # Check if the sequence is valid (not None)
        if sequence is not None:
            print(f"Sampled sequence at ({x}, {y}): {sequence}")
            sample_count += 1
            data.extend(sequence)
        else:
            print(f"Out of bounds at ({x}, {y})")

data = np.array(data)

print(f"Total samples: {sample_count}")
print(f"data shape: {data.shape}")

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
        # sequence_length by sequence_length data (n x n)
        return (torch.tensor(self.data[index * self.sequence_length : index * self.sequence_length + self.sequence_length - 1], dtype=torch.long),
                torch.tensor(self.data[index * self.sequence_length + 1 : index * self.sequence_length + self.sequence_length], dtype=torch.long))

dataset = TileMapDataset(data, sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%

embedding_dim = 8
hidden_dim = 16
n_layers = 1

class TileModelRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.0):
        super(TileModelRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, _ = self.rnn(embedded)
        
        output = self.fc(rnn_output)
        return output
    
model = TileModelRNN(vocabulary_size, embedding_dim, hidden_dim, vocabulary_size, n_layers).cuda()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}') # Number of parameters: 152405
# %%

learning_rate = 5e-4
num_epochs = 5000

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
for epoch in pbar:
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocabulary_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    # Update the progress bar's postfix with the average loss at the end of the epoch.
    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")


torch.save(model, f'model_rnn.pth')
# %%
PAD_TOKEN = 0
SIZE = 64
def generate_tilemap(model, ix, iy, length=16):
    generated_map = [PAD_TOKEN] * (length * length)

    def sample_map(x, y):
        if x < 0 or x >= length or y < 0 or y >= length:
            return PAD_TOKEN
        return generated_map[y * length + x]

    with torch.no_grad():
        for y in range(length):
            for x in range(length):
                row = []
                for i in range(-(sequence_length - 1), 0):
                    row.append(sample_map(x + i, y))

                input_tensor = torch.tensor(row, dtype=torch.long).unsqueeze(0).cuda()
                
                raw_output = model(input_tensor)
                raw_output = raw_output[0, -1]

                raw_output = F.softmax(raw_output, dim=-1)
                
                next_token = torch.multinomial(raw_output, 1).item()

                generated_map[y * length + x] = next_token
    
    return generated_map
                
model.eval()
tile_map_gen = generate_tilemap(model, 0, 0,SIZE)

tile_map_2d = np.array(tile_map_gen).reshape((SIZE, SIZE))
print(tile_map_2d)

# %%
# Folder where unique tile images are stored.
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
# %%


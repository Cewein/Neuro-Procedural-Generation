#!/usr/bin/env python3
"""
Example Script: Extracting, Merging Tiles, and Creating map.dat
------------------------------------------------------------------
This script demonstrates the following workflow:
  1. An image (representing many game tiles) is loaded.
  2. The image is divided into fixed-size tiles.
  3. Each tile is quantized to only a limited number of colors 
     (simulating a 2bpp format with a default of 4 colors).
  4. A hash (using MD5) is computed on the tile’s binary data.
  5. Graphically identical tiles (i.e. with the same hash) are merged.
  6. A mapping is built that records, for each tile in the original image,
     the integer ID of its corresponding unique tile.
  7. Each unique tile is saved as a separate image file.
  8. A binary file (map.dat) is saved that contains the tile mapping
     as 32-bit integer IDs.
     
Requirements: Pillow (PIL) and Python 3.
Usage:
  python extract_tiles.py --input overworld.png --tile_width 16 --tile_height 16
"""

import argparse
import os
import hashlib
from PIL import Image
import numpy as np

def quantize_tile(tile, colors=4):
    """
    Convert the tile image to palette mode with a limited number of colors.
    This simulates a 2bpp image where only 'colors' are available.
    """
    return tile.convert("P", palette=Image.ADAPTIVE, colors=colors)

def tile_hash(tile):
    """
    Compute an MD5 hash for the tile image data.
    """
    data = tile.tobytes()
    return hashlib.md5(data).hexdigest()

def extract_tiles(image, tile_width, tile_height):
    """
    Split the input image into tiles of (tile_width x tile_height).
    Returns a list of tile images.
    """
    tiles = []
    img_width, img_height = image.size
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            box = (x, y, x + tile_width, y + tile_height)
            tile = image.crop(box)
            tiles.append(tile)
    return tiles

def process_tiles(tiles, colors=4):
    """
    Process the list of tiles:
      - Quantize each tile to a fixed number of colors.
      - Compute a hash for each tile.
      - Build a dictionary of unique tiles (hash -> quantized tile image).
      - Build a mapping (list) where each entry is the tile’s hash.
    Returns the dictionary (unique_tiles) and the mapping (list of hashes).
    """
    unique_tiles = {}  # hash -> quantized tile image
    mapping = []       # one entry per input tile: the tile's hash
    for tile in tiles:
        quant_tile = quantize_tile(tile, colors=colors)
        h = tile_hash(quant_tile)
        if h not in unique_tiles:
            unique_tiles[h] = quant_tile
        mapping.append(h)
    return unique_tiles, mapping

def save_unique_tiles(unique_tiles, output_dir):
    """
    Save each unique tile from the dictionary as a separate PNG file.
    Each unique tile is assigned a numeric ID based on its order.
    Returns a dictionary mapping tile hash to its numeric ID.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tile_id_mapping = {}
    for idx, (h, tile) in enumerate(unique_tiles.items()):
        tile_id_mapping[h] = idx
        filename = os.path.join(output_dir, f"unique_tile_{idx}.png")
        tile.save(filename)
        print(f"Saved unique tile {idx} (hash {h}) to {filename}")
    return tile_id_mapping

def save_tile_mapping_text(mapping, tile_id_mapping, output_file):
    """
    Save the tile mapping to a text file.
    Each line corresponds to one tile in the original image and contains the
    unique integer tile ID.
    """
    with open(output_file, "w") as f:
        for i, h in enumerate(mapping):
            unique_id = tile_id_mapping[h]
            f.write(f"{i}: {unique_id}\n")
    print(f"Tile mapping (text) saved to {output_file}")

def save_map_dat(mapping, tile_id_mapping, output_file):
    """
    Convert the tile mapping from hashes to numeric IDs and save it
    as a binary file (map.dat) using 32-bit integers.
    """
    numeric_mapping = [tile_id_mapping[h] for h in mapping]
    mapping_array = np.array(numeric_mapping, dtype=np.int32)
    mapping_array.tofile(output_file)
    print(f"Binary tile mapping saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract and merge tiles from an image and create map.dat")
    parser.add_argument('--input', type=str, required=True, help="Input image file (e.g. overworld.png)")
    parser.add_argument('--tile_width', type=int, default=16, help="Tile width in pixels (default: 16)")
    parser.add_argument('--tile_height', type=int, default=16, help="Tile height in pixels (default: 16)")
    parser.add_argument('--output_dir', type=str, default="unique_tiles", help="Directory to save unique tile images")
    parser.add_argument('--mapping_file', type=str, default="tile_mapping.txt", help="Output text file for tile mapping")
    parser.add_argument('--map_file', type=str, default="map.dat", help="Output binary file for tile mapping")
    parser.add_argument('--colors', type=int, default=4, help="Number of colors for quantization (default: 4)")
    args = parser.parse_args()

    # Load input image.
    try:
        image = Image.open(args.input)
    except Exception as e:
        print(f"Error opening image {args.input}: {e}")
        return

    print(f"Loaded image {args.input} of size {image.size}")

    # Extract tiles.
    tiles = extract_tiles(image, args.tile_width, args.tile_height)
    print(f"Extracted {len(tiles)} tiles (each {args.tile_width}x{args.tile_height}px)")

    # Process tiles: quantize, compute hash, merge duplicates, and build mapping.
    unique_tiles, mapping = process_tiles(tiles, colors=args.colors)
    print(f"Found {len(unique_tiles)} unique tiles after merging duplicates")

    # Save the unique tile images to disk.
    tile_id_mapping = save_unique_tiles(unique_tiles, args.output_dir)

    # Save the mapping as a text file containing only integer IDs.
    save_tile_mapping_text(mapping, tile_id_mapping, args.mapping_file)

    # Save the mapping as a binary file "map.dat" (each tile mapped to a numeric ID).
    save_map_dat(mapping, tile_id_mapping, args.map_file)

if __name__ == '__main__':
    main()

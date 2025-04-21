#!/usr/bin/env python3
"""
Example Script: Extracting, Merging Tiles, and Creating map.dat from a Picture or a Folder of Pictures
-----------------------------------------------------------------------------------------------
This script demonstrates the following workflow for each input image or folder:
  1. Load an image (or images from a folder).
  2. Divide the image(s) into fixed-size tiles.
  3. (Optionally) Quantize each tile to a limited number of colors.
  4. Compute an MD5 hash on each tile’s binary data.
  5. Merge graphically identical tiles based on their hash.
  6. Build a mapping that records for each original tile the unique tile’s integer ID.
  7. Save each unique tile as a separate image file (shared across all images in a folder).
  8. Write a text mapping file and a binary file (map.dat) with the tile mapping (32-bit integer IDs).

Requirements: Pillow (PIL), NumPy, and Python 3.
Usage:
  Process a single image:
    python extract_tiles.py --input overworld.png --tile_width 16 --tile_height 16

  Process all images in a folder (shared unique tiles):
    python extract_tiles.py --input folder_of_images --tile_width 16 --tile_height 16

Notes:
  The script supports common image file extensions (png, jpg, jpeg, bmp, gif).
"""

import argparse
import os
import sys
import hashlib
from PIL import Image
import numpy as np

# List of valid image file extensions (case insensitive)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

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
      - Optionally quantize each tile to a fixed number of colors.
      - Compute a hash for each tile.
      - Build a dictionary of unique tiles (hash -> tile image).
      - Build a mapping (list) where each entry is the tile’s hash.
    Returns the dictionary (unique_tiles) and the mapping (list of hashes).
    """
    unique_tiles = {}  # hash -> tile image
    mapping = []       # one entry per input tile: the tile's hash
    for tile in tiles:
        # Uncomment the next line if you want to apply quantization:
        # tile = quantize_tile(tile, colors=colors)
        h = tile_hash(tile)
        if h not in unique_tiles:
            unique_tiles[h] = tile
        mapping.append(h)
    return unique_tiles, mapping

def save_unique_tiles(unique_tiles, output_dir, width=16, height=16):
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
        filename = os.path.join(output_dir, f"unique_tile_{width}-{height}_{idx}.png")
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

def process_image_file(image_path, tile_width, tile_height, colors, output_base, mapping_filename, binary_filename):
    """
    Process a single image file (legacy behavior):
      - Creates its own subfolder under output_base.
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    print(f"Processing image '{image_path}' (size: {image.size})")
    tiles = extract_tiles(image, tile_width, tile_height)
    print(f"Extracted {len(tiles)} tiles ({tile_width}x{tile_height}px each) from '{image_path}'")
    unique_tiles, mapping = process_tiles(tiles, colors=colors)
    print(f"Found {len(unique_tiles)} unique tiles in '{image_path}'")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_base, base_name)
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    tile_id_mapping = save_unique_tiles(unique_tiles, image_output_dir)

    text_mapping_path = os.path.join(image_output_dir, mapping_filename)
    save_tile_mapping_text(mapping, tile_id_mapping, text_mapping_path)

    binary_mapping_path = os.path.join(image_output_dir, binary_filename)
    save_map_dat(mapping, tile_id_mapping, binary_mapping_path)

def main():
    parser = argparse.ArgumentParser(description="Extract and merge tiles from an image or a folder of images and create map.dat")
    parser.add_argument('--input', type=str, required=True, help="Input image file or folder of image files (supported formats: png, jpg, jpeg, bmp, gif)")
    parser.add_argument('--tile_width', type=int, default=16, help="Tile width in pixels (default: 16)")
    parser.add_argument('--tile_height', type=int, default=16, help="Tile height in pixels (default: 16)")
    parser.add_argument('--output_dir', type=str, default="output_tiles", help="Directory to save output (unique tile images and mapping files)")
    parser.add_argument('--mapping_file', type=str, default="tile_mapping.txt", help="Suffix for text mapping files (per image)")
    parser.add_argument('--map_file', type=str, default="map.dat", help="Suffix for binary mapping files (per image)")
    parser.add_argument('--colors', type=int, default=4, help="Number of colors for quantization (default: 4). Quantization is optional in processing.")
    args = parser.parse_args()

    input_path = args.input

    if os.path.isdir(input_path):
        # Aggregate all images in the folder, sharing unique tiles
        files = sorted(os.listdir(input_path))
        image_files = [os.path.join(input_path, f) for f in files if f.lower().endswith(VALID_EXTENSIONS)]
        if not image_files:
            print(f"No valid image files found in directory: {input_path}")
            sys.exit(1)
        print(f"Found {len(image_files)} image file(s) in folder '{input_path}'.")

        global_unique_tiles = {}
        image_mappings = {}

        # Extract and merge tiles across all images
        total_tiles = 0
        for img_file in image_files:
            try:
                image = Image.open(img_file)
            except Exception as e:
                print(f"Error opening image {img_file}: {e}")
                continue
            tiles = extract_tiles(image, args.tile_width, args.tile_height)
            total_tiles += len(tiles)
            mapping = []
            for tile in tiles:
                # Uncomment to quantize:
                # tile = quantize_tile(tile, colors=args.colors)
                h = tile_hash(tile)
                if h not in global_unique_tiles:
                    global_unique_tiles[h] = tile
                mapping.append(h)
            image_mappings[img_file] = mapping

        print(f"Extracted a total of {total_tiles} tiles across all images.")
        print(f"Found {len(global_unique_tiles)} unique tiles across folder '{input_path}'.")

        # Save all unique tiles once
        tile_id_mapping = save_unique_tiles(global_unique_tiles, args.output_dir, args.tile_width, args.tile_height)

        # Save mapping files for each image
        for img_file, mapping in image_mappings.items():
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            
            image_output_dir = os.path.join(args.output_dir, "text/")
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            bin_output_dir = os.path.join(args.output_dir, "bin/")
            if not os.path.exists(bin_output_dir):
                os.makedirs(bin_output_dir)

            text_path = os.path.join(image_output_dir, f"{base_name}_{args.mapping_file}")
            save_tile_mapping_text(mapping, tile_id_mapping, text_path)
            bin_path = os.path.join(bin_output_dir, f"{base_name}_{args.map_file}")
            save_map_dat(mapping, tile_id_mapping, bin_path)

    elif os.path.isfile(input_path):
        # Process a single image file (legacy behavior)
        process_image_file(input_path, args.tile_width, args.tile_height, args.colors,
                           args.output_dir, args.mapping_file, args.map_file)
    else:
        print(f"Error: The input '{input_path}' is neither a file nor a directory.")
        sys.exit(1)

if __name__ == '__main__':
    main()

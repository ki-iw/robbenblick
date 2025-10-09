import os
from PIL import Image
import math
import glob

# def tile_images(
#     image_path,
#     output_dir,
#     tile_size=1024,
#     overlap=0,
#     output_format="tif"
# ):
#     """
#     Splits a large .tif image into tiles.

#     Args:
#         image_path (str): Path to input .tif image.
#         output_dir (str): Directory where tiles will be saved.
#         tile_size (int): Size of each tile (square, in pixels).
#         overlap (int): Number of pixels to overlap between tiles.
#         output_format (str): 'tif' or 'png'.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Load image
#     img = Image.open(image_path)
#     width, height = img.size
#     print(f"Loaded image size: {width}x{height}")

#     # Calculate number of tiles
#     x_tiles = math.ceil((width - overlap) / (tile_size - overlap))
#     y_tiles = math.ceil((height - overlap) / (tile_size - overlap))

#     print(f"Tiling into {x_tiles} x {y_tiles} = {x_tiles*y_tiles} tiles")

#     for i in range(x_tiles):
#         for j in range(y_tiles):
#             left = i * (tile_size - overlap)
#             upper = j * (tile_size - overlap)
#             right = min(left + tile_size, width)
#             lower = min(upper + tile_size, height)

#             # Crop tile
#             tile = img.crop((left, upper, right, lower))

#             # Pad if tile is smaller than tile_size
#             if tile.size != (tile_size, tile_size):
#                 padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
#                 padded.paste(tile, (0, 0))
#                 tile = padded

#             # Save tile
#             tile_filename = f"tile_{i}_{j}.{output_format}"
#             tile.save(os.path.join(output_dir, tile_filename))

def tile_images(source_dir, dest_dir, tile_size, overlap_percent):
    """
    Tiles all .tif images from a source directory into smaller, overlapping tiles.

    Args:
        source_dir (str): The directory containing the original .tif images.
        dest_dir (str): The directory where the new tiles will be saved.
        tile_size (tuple): The (width, height) of the tiles in pixels.
        overlap_percent (int): The percentage of overlap between tiles (0-100).
    """

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    # Calculate overlap in pixels
    overlap_pixels_x = int(tile_size[0] * (overlap_percent / 100))
    overlap_pixels_y = int(tile_size[1] * (overlap_percent / 100))

    # Get a list of all .tif files in the source directory
    image_files = glob.glob(os.path.join(source_dir, "*.TIF"))
    if not image_files:
        print(f"No .tif images found in {source_dir}. Exiting.")
        return

    print(f"Found {len(image_files)} images to tile.")
    
    # Iterate through each image and tile it
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img_width, img_height = img.size

                print(f"Processing '{img_name}.tif' ({img_width}x{img_height} pixels)...")

                tile_count = 0
                for y in range(0, img_height, tile_size[1] - overlap_pixels_y):
                    for x in range(0, img_width, tile_size[0] - overlap_pixels_x):
                        # Define the crop box
                        left = x
                        upper = y
                        right = x + tile_size[0]
                        lower = y + tile_size[1]

                        # Adjust the box if it extends beyond the image boundaries
                        if right > img_width:
                            right = img_width
                            left = right - tile_size[0]
                        if lower > img_height:
                            lower = img_height
                            upper = lower - tile_size[1]
                        
                        # Crop the image
                        tile = img.crop((left, upper, right, lower))
                        
                        # Save the new tile
                        tile_filename = f"{img_name}_tile_{tile_count}.tif"
                        tile_path = os.path.join(dest_dir, tile_filename)
                        tile.save(tile_path)
                        tile_count += 1
                
                print(f"Successfully created {tile_count} tiles for '{img_name}.tif'.")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
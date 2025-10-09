
import os
from PIL import Image

from robbenblick import logger

def convert_tif_to_png(src_folder, dst_folder):
    """
    Converts all .tif images in src_folder to .png and saves them in dst_folder.
    Args:
        src_folder (str): Source folder containing .tif images.
        dst_folder (str): Destination folder to save .png images.
    """
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if fname.lower().endswith('.tif'):
            tif_path = os.path.join(src_folder, fname)
            png_name = os.path.splitext(fname)[0] + '.png'
            png_path = os.path.join(dst_folder, png_name)
            try:
                with Image.open(tif_path) as im:
                    im.save(png_path)
            except Exception as e:
                logger.error(f"Error converting {tif_path}: {e}")

def convert_xml_tif_to_png(xml_path,xml_path_png):
    """
    Replaces all occurrences of .tif with .png in the given XML file.
    Args:
        xml_path (str): Path to the XML file to modify.
    """
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('.tif', '.png')
    xml_content = xml_content.replace('.TIF', '.png')
    with open(xml_path_png, 'w') as f:
        f.write(xml_content)
    
import os
from image_info import get_image_size
from PIL import Image

MIN_WIDTH = 392
MIN_HEIGHT = 696
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
# Height / width of the training input; landscape images are padded onto a
# canvas of this aspect ratio instead of being rotated, so their real
# orientation is preserved.
TARGET_ASPECT = MIN_HEIGHT / MIN_WIDTH


def pad_to_portrait(image_file):
    with Image.open(image_file) as im:
        im = im.convert("RGB")
        width, height = im.size
        target_height = round(width * TARGET_ASPECT)
        if target_height <= height:
            return None
        canvas = Image.new("RGB", (width, target_height))
        canvas.paste(im, (0, (target_height - height) // 2))
    out_name = os.path.splitext(image_file)[0] + "_padded.png"
    canvas.save(out_name)
    os.remove(image_file)
    return out_name


for dirpath, _, filenames in os.walk("screenshots"):
    image_names = [name for name in filenames if name.lower().endswith(IMAGE_EXTENSIONS)]
    if len(image_names) < 2:
        print("TOO SMALL: " + dirpath)
        continue
    for image in image_names:
        image_file = os.path.join(dirpath, image)
        width, height = get_image_size(image_file)
        if width is None or height is None:
            print("UNREADABLE: " + image_file)
            continue
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            print("Small image: " + image_file + " - " + str(width) + " x " + str(height))
        if width > height:
            print("WIDE: " + image_file)
            try:
                out_name = pad_to_portrait(image_file)
                if out_name:
                    print("saving: " + out_name)
            except (OSError, ValueError) as err:
                print("FAILED padding " + image_file + ": " + str(err))

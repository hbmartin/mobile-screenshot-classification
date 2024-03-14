import os
from image_info import get_image_size
from PIL import Image

for dirname in os.walk("screenshots"):
    if len(dirname[2]) < 2:
        print("TOO SMALL: " + dirname[0])
        continue
    for image in dirname[2]:
        if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
            image_file = dirname[0] + "/" + image
            image_size = get_image_size(image_file)
            if image_size[0] < 392 or image_size[1] < 696:
                print("Small image: " + image_file + " - " + str(image_size[0]) + " x " + str(image_size[1]))
            if image_size[0] > image_size[1]:
                print("WIDE: " + image_file)
                try:
                    im = Image.open(image_file)
                    out = im.rotate(90, expand=True)
                    out_name = ".".join(image_file.split(".")[:-1]) + "_90.png"
                    print("saving: " + out_name)
                    out.save(out_name)
                    os.remove(image_file)
                except:
                    print("FAILED rotating above")
        else:
            print("Not processing: " + image)
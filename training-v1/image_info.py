from io import BytesIO
import struct
from typing import Optional

gif_types = (bytes("GIF87a", "utf-8"), bytes("GIF89a", "utf-8"))
png_type = bytes("\x89PNG\r\n\x1A\n", "utf-8")[1:]
ihdr = bytes("IHDR", "utf-8")
jpeg_type = b"\xFF\xD8"


def get_image_size(file_path: str) -> tuple[Optional[int], Optional[int]]:
    file_handle = open(file_path, "rb")
    data = file_handle.read()
    file_handle.close()

    size = len(data)
    height: Optional[int] = None
    width: Optional[int] = None
    content_type = ''

    # handle GIFs
    if (size >= 10) and data[:6] in gif_types:
        # Check to see if content_type is correct
        content_type = 'image/gif'
        w, h = struct.unpack("<HH", data[6:10])
        width = int(w)
        height = int(h)

    # See PNG 2. Edition spec (http://www.w3.org/TR/PNG/)
    # Bytes 0-7 are below, 4-byte chunk length, then 'IHDR'
    # and finally the 4-byte width, height
    elif ((size >= 24) and data.startswith(png_type)
          and (data[12:16] == ihdr)):
        content_type = 'image/png'
        w, h = struct.unpack(">LL", data[16:24])
        width = int(w)
        height = int(h)

    # Maybe this is for an older PNG version.
    elif (size >= 16) and data.startswith(png_type):
        # Check to see if we have the right content type
        content_type = 'image/png'
        w, h = struct.unpack(">LL", data[8:16])
        width = int(w)
        height = int(h)

    # handle JPEGs
    elif (size >= 2) and data.startswith(jpeg_type):
        content_type = 'image/jpeg'
        jpeg = BytesIO(data)
        jpeg.read(2)
        b = jpeg.read(1)
        try:
            while (b and ord(b) != 0xDA):
                while (ord(b) != 0xFF): b = jpeg.read(1)
                while (ord(b) == 0xFF): b = jpeg.read(1)
                if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                    jpeg.read(3)
                    h, w = struct.unpack(">HH", jpeg.read(4))
                    break
                else:
                    jpeg.read(int(struct.unpack(">H", jpeg.read(2))[0])-2)
                b = jpeg.read(1)
            width = int(w)
            height = int(h)
        except struct.error:
            pass
        except ValueError:
            pass

    return width, height

from itertools import compress

import exif


def filter_by_defocus_sign(list_images: list[str], positive_sign: bool) -> list[str]:
    list_index = []
    for img_path in list_images:
        if positive_sign:
            if float(exif.Image(img_path).make) >= 0:
                list_index.append(True)
            else:
                list_index.append(False)

        else:
            if float(exif.Image(img_path).make) <= 0:
                list_index.append(True)
            else:
                list_index.append(False)

    return list(compress(list_images, list_index))
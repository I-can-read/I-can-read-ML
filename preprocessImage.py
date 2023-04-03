from PIL import Image
import cv2
import numpy as np
import easyocr
from fastapi import UploadFile


def gray_scale(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return result


def preprocess_image(file: UploadFile):
    contents = file.file.read()
    img = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    # img = cv2.imdecode(np.fromstring(contents, np.unit8), cv2.IMREAD_COLOR)
    # img = cv2.imread(file, cv2.IMREAD_COLOR)
    image_gray = gray_scale(img)
    height, width = image_gray.shape
    gray_enlarge = cv2.resize(image_gray, (4 * width, 4 * height), interpolation=cv2.INTER_LINEAR)

    denoised = cv2.fastNlMeansDenoising(gray_enlarge, h=10, searchWindowSize=21, templateWindowSize=7)
    max_output_value = 255
    neighborhood_size = 125
    subtract_from_mean = 9
    image_binarized = cv2.adaptiveThreshold(denoised, max_output_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, neighborhood_size, subtract_from_mean)

    dst = "./menu.png"
    completedimage = image_binarized
    cv2.imwrite(dst, completedimage)
    completedimage = "./menu.png"
    return completedimage


def crop_image(image):
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(image)

    img = cv2.imread(image)
    img = Image.fromarray(img)

    count = 0
    name = []
    for i in range(len(result)):
        name.append(i)

    for i in result:
        x = i[0][0][0]
        y = i[0][0][1]
        w = i[0][1][0] - i[0][0][0]
        h = i[0][2][1] - i[0][1][1]
        image_name = i[1]

        comma = ','
        point = '.'

        if (comma in image_name):
            image_name = image_name.replace(",", "")
        if (point in image_name):
            image_name = image_name.replace(".", "")
        tf = image_name.isdigit()
        if (not tf and (not image_name.encode().isalpha())):
            cropped_image = img.crop((x, y, x + w, y + h))
            cropped_image.save(f'./words/{name[count]}.png')
            count = count + 1


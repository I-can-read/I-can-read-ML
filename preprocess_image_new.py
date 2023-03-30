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

    results = []
    results.clear()

    for i in result:
        text = i[1]
        comma = ','
        point = '.'

        if (comma in text):
            text = text.replace(",", "")

        if (point in text):
            text = text.replace(".", "")

        tf = text.isdigit()

        if (not tf and (not text.encode().isalpha())):
            results.append(text)

    return results

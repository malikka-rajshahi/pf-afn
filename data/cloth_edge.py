import cv2
import numpy as np
import os
import time

class GenerateEdge:
    def __init__(self, cloth_path):
        print('Running cloth mask')
        self.cloth_path = cloth_path
    

    def get_cloth_mask(self):
        image = cv2.imread(self.cloth_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 2:
            contours, _ = contours
        elif len(contours) == 3:
            _, contours, _ = contours

        mask = np.zeros_like(image)

        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

        return mask

    def process_images(self, output_path):
        cloth_mask = GenerateEdge.get_cloth_mask(self)[:, :, 0]

        cv2.imwrite(output_path, cloth_mask)

        print(f"Cloth mask saved at: {output_path}")

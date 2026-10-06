import cv2
import numpy as np

image = np.zeros((200, 200), dtype=np.uint8)
image[30:100, 30:100] = 180
image[120:180, 120:180] = 220
noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

blur = cv2.GaussianBlur(noisy_image, (5, 5), 0)
otsu_threshold, thresholded_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Computed Otsu's Threshold Value: {otsu_threshold}")
print(f"Original Image Shape: {image.shape}")
print(f"Thresholded Image Unique Values: {np.unique(thresholded_img)}")

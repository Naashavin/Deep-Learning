import cv2
import numpy as np

img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(img, (70, 70), 40, (255, 255, 255), -1)
cv2.circle(img, (120, 120), 40, (255, 255, 255), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

num_segmented_objects = len(np.unique(markers)) - 2
print("Watershed Segmentation Complete.")
print("Unique markers detected:", np.unique(markers))
print("Number of segmented regions:", num_segmented_objects)

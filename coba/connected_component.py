import cv2
import numpy as np

image = cv2.imread('C:/Users/Rendicahya/Desktop/segmentation.jpg', 0)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
sizes = stats[1:, -1]
nb_components -= 1
filtered = np.zeros(image.shape)

print(nb_components)
print(stats.shape)

for i in range(0, nb_components):
    if sizes[i] >= 10:
        filtered[output == i + 1] = 255

cv2.imshow('Image', image)
cv2.imshow('Filtered', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()

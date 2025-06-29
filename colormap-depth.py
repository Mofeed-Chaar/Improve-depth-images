import cv2
import numpy as np
import os
# Load using OpenCV preserving depth (same as 'I' mode)
img = cv2.imread('/home/chaar/Desktop/extended/img/draft/test/munster_000000_000000_disparity.png', cv2.IMREAD_UNCHANGED)
print(img.dtype)
# Normalize to 8-bit
img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
img_8bit = img_normalized.astype(np.uint8)

# Apply colormap
img_colored = cv2.applyColorMap(img_8bit, cv2.COLORMAP_JET)

# Save result
cv2.imwrite('/home/chaar/Desktop/extended/img/draft/test/completed.png', img_colored)
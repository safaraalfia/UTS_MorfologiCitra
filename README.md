# UTS_MorfologiCitra
Alfia Rohmah Safara (23422039)
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.naive_bayes import GaussianNB

# Gambar yang telah di-upload
img = cv2.imread('flower.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold dan morfologi
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Labeling dan ekstraksi fitur bentuk
label_img = label(morph)
props = regionprops(label_img)

# Ekstraksi fitur bentuk dan warna
for prop in props:
    if prop.area > 500:  # Filter objek kecil
        area = prop.area
        perimeter = prop.perimeter
        aspect_ratio = prop.bbox_area / area
        mean_color = np.mean(img_rgb.reshape(-1, 3), axis=0)

        print(f"Luas: {area}, Keliling: {perimeter:.2f}, Rasio: {aspect_ratio:.2f}")
        print(f"Rata-rata Warna RGB: {mean_color}")

        # Klasifikasi dummy
        X_train = [[3000, 100, 1.5, 120, 100, 90]]
        y_train = ['mawar']
        X_test = [[area, perimeter, aspect_ratio, *mean_color]]
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        print("Kelas bunga yang diprediksi:", pred[0])

# Tampilkan hasil
plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title("Asli")
plt.subplot(1,3,2); plt.imshow(morph, cmap='gray'); plt.title("Segmentasi Morfologi")
plt.subplot(1,3,3); plt.imshow(label_img, cmap='nipy_spectral'); plt.title("Labeling")
plt.show()
```

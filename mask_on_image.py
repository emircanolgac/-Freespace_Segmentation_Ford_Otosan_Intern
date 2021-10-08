import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *

# Tüm maskeleri içeren bir liste oluşturuldu
mask_list = os.listdir(MASK_DIR)

# Eğer gizli dosya varsa listeden çıkarıldı
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

for mask_name in tqdm.tqdm(mask_list):
    # maske isimlerinden uzantıları çıkarıldı
    mask_name_without_ex = mask_name.split('.')[0]

    # maske, resim ve maskelenmiş resimlerin kaydedileceği dosya yolu ilgili değişkenlere atandı
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name_without_ex+'.jpg')
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # maske ve ilgili orjinal resim okundu
    mask  = cv2.imread(mask_path, 0).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)

    # orijinal görüntüdeki maske kısmına karşılık gelen piksellerin rengi değiştirildi ve yeni görüntü oluşturuldu
    cpy_image  = image.copy()
    image[mask==1, :] = (255, 0, 125)
    opac_image = (image/2 + cpy_image/2).astype(np.uint8)

    # yeni görüntü ilgili dosyaya kaydedildi
    cv2.imwrite(image_out_path, opac_image)

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:
        plt.figure()
        plt.imshow(opac_image)
        plt.show()

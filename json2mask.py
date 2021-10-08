import json
import os
import numpy as np
import cv2
import tqdm
from constant import JSON_DIR, MASK_DIR

# Her json dosyası json_list listesine eklendi
json_list = os.listdir(JSON_DIR)

for json_name in tqdm.tqdm(json_list):

    # json dosyaları okundu
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # json dosyaları yüklendi
    json_dict = json.load(json_file)

    # Orjinal resimler ile aynı boyutta boş bir maske oluşturuldu
    mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    mask_path = os.path.join(MASK_DIR, json_name[:-9]+".png")

    for obj in json_dict["objects"]:
        # 'classTitle'ın 'Freespace'e eşit olup olmadığı kontrol edildi
        if obj['classTitle']=='Freespace':
            # point'lerin bulunduğu konumlara göre boş maske görüntüsü dolduruldu
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    # Doldrulan maske görüntüleri ilgili dosyaya kaydedildi
    cv2.imwrite(mask_path, mask.astype(np.uint8))

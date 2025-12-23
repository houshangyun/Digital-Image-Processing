import tensorflow as tf
import numpy as np
import cv2
import json

IMG_SIZE = 224

# 載入模型 & 類別
model = tf.keras.models.load_model("garbage_mobilenetv2.h5")
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# 反轉 {label:index} → {index:label}
idx_to_class = {v: k for k, v in class_indices.items()}

# 輸入圖片
image_path = input("Enter image path: ")
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_norm = img_resized.astype("float32") / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# 預測
pred = model.predict(img_input)
prob = np.max(pred)
label = idx_to_class[np.argmax(pred)]

# 顯示文字在圖片上方
text = f"{label} ({prob:.2f})"
cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 255, 0), 2)

cv2.imshow("Prediction", img)
cv2.waitKey(0)

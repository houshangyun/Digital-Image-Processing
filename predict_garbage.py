import tensorflow as tf
import numpy as np
import cv2
import json
import os
import csv

IMG_SIZE = 224

# 載入模型與類別
model = tf.keras.models.load_model("garbage_mobilenetv2.h5")
with open("classes.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# 指定要預測的資料集資料夾
dataset_dir = input("Enter dataset folder path: ")

# 建立輸出結果 CSV
output_csv = "prediction_results.csv"
csv_file = open(output_csv, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Image_Path", "True_Label", "Predicted_Label", "Probability"])

# 統計用
total = 0
correct = 0

# 遞迴走訪所有子資料夾與圖片
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)

            # Ground Truth = 資料夾名稱
            true_label = os.path.basename(root)

            # 載入圖片
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_norm = img_resized.astype("float32") / 255.0
            img_input = np.expand_dims(img_norm, axis=0)

            # 預測
            pred = model.predict(img_input, verbose=0)
            prob = np.max(pred)
            pred_label = idx_to_class[np.argmax(pred)]

            # 統計
            total += 1
            if pred_label == true_label:
                correct += 1

            # 寫入 CSV
            csv_writer.writerow([img_path, true_label, pred_label, f"{prob:.4f}"])

            # 終端顯示
            print(f"{img_path} → TRUE: {true_label} | PRED: {pred_label} ({prob:.2f})")

csv_file.close()

# 計算準確率
accuracy = correct / total if total > 0 else 0
print("\n===============================")
print(f"📊 Overall Accuracy: {accuracy * 100:.2f}%")
print("===============================")
print("Results saved to:", output_csv)

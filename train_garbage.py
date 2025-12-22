import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 路徑：換成你自己的資料夾
DATASET_DIR = "/home/yoon/Downloads/Garbage classification/"

# 參數
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15   # CPU 建議 10~20，越高越準

# 1️⃣ 資料前處理與增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 2️⃣ 載入 MobileNetV2（不包含頂層）
base_model = MobileNetV2(weights='imagenet', include_top=False,
                         input_shape=(IMG_SIZE, IMG_SIZE, 3))

base_model.trainable = False  # 先凍結特徵Extractor，加速訓練

# 3️⃣ 建立分類頭
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# 4️⃣ 編譯模型
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5️⃣ 開始訓練
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ===============================
# ⭐ 加入混淆矩陣（放在這裡）
# ===============================
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("[INFO] generating confusion matrix...")

# 取得驗證資料的真實標籤
val_generator.reset()
valX, valY = [], []

for _ in range(len(val_generator)):
    x_batch, y_batch = val_generator.next()
    valX.append(x_batch)
    valY.append(y_batch)

valX = np.vstack(valX)
valY = np.vstack(valY)

# 模型預測
y_pred = model.predict(valX)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(valY, axis=1)

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred_labels)

classNames = list(train_generator.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=classNames))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classNames,
            yticklabels=classNames)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ===============================
# 6️⃣ 儲存模型
# ===============================
model.save("garbage_mobilenetv2.h5")
print("✅ 訓練完成，模型已儲存為 garbage_mobilenetv2.h5")

# 7️⃣ 儲存類別名稱
import json
with open("classes.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("📄 類別索引已儲存為 classes.json")

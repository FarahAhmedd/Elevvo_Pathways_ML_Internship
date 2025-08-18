import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Image-based (Spectrogram) classification with CNN
import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
VGG16 = tf.keras.applications.VGG16
layers = tf.keras.layers
models = tf.keras.models

# Tabular (MFCC) classification
mfcc_df = pd.read_csv('Datasets/Data/features_30_sec.csv')

# Remove .wav extension from filename column if present
mfcc_df['filename'] = mfcc_df['filename'].str.replace('.wav', '', regex=False)

X_tab = mfcc_df.drop(['label', 'filename'], axis=1)
y_tab = mfcc_df['label']

le = LabelEncoder()
y_tab_enc = le.fit_transform(y_tab)
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tab_scaled, y_tab_enc, test_size=0.2, random_state=42, stratify=y_tab_enc)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_t, y_train_t)
y_pred_t = clf.predict(X_test_t)

print("Tabular (MFCC) Classification Report:")
print(classification_report(y_test_t, y_pred_t, target_names=le.classes_))

# Assume spectrogram images are in Datasets/Data/spectrograms/<genre>/<filename>.png
img_dir = 'Datasets/Data/images_original'
img_height, img_width = 128, 128
batch_size = 32

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)
val_gen = datagen.flow_from_directory(
    img_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Simple CNN
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(train_gen, validation_data=val_gen, epochs=10)

# Evaluate CNN
val_gen.reset()
loss, acc = cnn.evaluate(val_gen)
print(f"Image-based CNN Accuracy: {acc:.4f}")

# Transfer learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

tl_model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])
tl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tl_model.fit(train_gen, validation_data=val_gen, epochs=5)

# Evaluate transfer learning model
val_gen.reset()
loss_tl, acc_tl = tl_model.evaluate(val_gen)
print(f"Image-based Transfer Learning (VGG16) Accuracy: {acc_tl:.4f}")

# Compare results
print("\nComparison:")
print("Tabular (MFCC) Accuracy:", np.mean(y_pred_t == y_test_t))
print("Image-based CNN Accuracy:", acc)
print("Image-based Transfer Learning (VGG16) Accuracy:", acc_tl)
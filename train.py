import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# ------------------------------
# Custom Callback to stop at target accuracy
# ------------------------------
class TargetAccuracy(Callback):
    def __init__(self, target=0.95):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get("val_accuracy")
        if acc and acc >= self.target:
            print(f"\nReached {self.target*100:.0f}% validation accuracy. Stopping training.")
            self.model.stop_training = True

# ------------------------------
# Load dataset
# ------------------------------
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "fruits/train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    "fruits/validation",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# ------------------------------
# Save class names
# ------------------------------
class_names = train_ds_raw.class_names
print("Class names:", class_names)

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

# ------------------------------
# Data augmentation
# ------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

# ------------------------------
# Preprocessing with AUTOTUNE
# ------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds_raw.map(lambda x, y: (preprocess_input(x), y))

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------
# Load pre-trained MobileNetV2
# ------------------------------
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

base_model.trainable = False  # initially freeze

# ------------------------------
# Build model
# ------------------------------
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------
# Initial training (feature extraction)
# ------------------------------
print("Starting feature extraction training...")
model.fit(train_ds,
          validation_data=val_ds,
          epochs=20,
          callbacks=[
              TargetAccuracy(0.95),
              EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
              ModelCheckpoint("fruit_classifier_best.keras", save_best_only=True)
          ])

# ------------------------------
# Fine-tuning
# ------------------------------
print("Starting fine-tuning...")
base_model.trainable = True

# Freeze first layers, fine-tune deeper layers
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50,
          callbacks=[
              TargetAccuracy(0.95),
              EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
              ModelCheckpoint("fruit_classifier_best.keras", save_best_only=True)
          ])

# ------------------------------
# Save final model
# ------------------------------
model.save("fruit_classifier_finetuned.keras")
print("Final model saved as fruit_classifier_finetuned.keras")

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ==============================
# Dataset Configuration
# ==============================

DATASET_PATH = r"D:\DL dataset\gaussian_filtered_images"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ==============================
# Load Training Dataset
# ==============================

train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=SEED
)


# ==============================
# Load Validation Dataset
# ==============================

validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=SEED
)


# ==============================
# Class Names
# ==============================

class_names = train_dataset.class_names

print("Classes:")
print(class_names)


# ==============================
# Normalization
# Convert pixel values:
# 0-255  --->  0-1
# ==============================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)


# ==============================
# Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ]
)


# ==============================
# Apply Preprocessing
# ==============================

train_dataset = train_dataset.map(
    lambda images, labels:
    (data_augmentation(normalization_layer(images)), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==============================
# Performance Optimization
# ==============================

AUTOTUNE = tf.data.AUTOTUNE


train_dataset = (
    train_dataset
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=AUTOTUNE)
)


validation_dataset = (
    validation_dataset
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)


# ==============================
# Testing Dataset Pipeline
# ==============================

if __name__ == "__main__":

    print("\nDataset Verification")
    print("-------------------")

    print("Class Names:")
    print(class_names)

    for images, labels in train_dataset.take(1):

        print("\nImage Batch Shape:")
        print(images.shape)

        print("\nLabel Batch Shape:")
        print(labels.shape)
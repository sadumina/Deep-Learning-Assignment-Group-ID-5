import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


# ==============================
# Model Configuration
# ==============================

IMG_SIZE = (224, 224)
NUM_CLASSES = 5


# ==============================
# Create EfficientNetB0 Model
# ==============================

def create_efficientnetb0():

    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )


    # Freeze pretrained layers
    base_model.trainable = False


    # Create classification model

    model = models.Sequential(
        [
            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dropout(0.3),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )
        ]
    )


    return model



# ==============================
# Model Summary Test
# ==============================

if __name__ == "__main__":

    model = create_efficientnetb0()

    model.summary()
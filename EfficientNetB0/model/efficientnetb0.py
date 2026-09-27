import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


IMG_SIZE = (224,224)
NUM_CLASSES = 5


def create_efficientnetb0():

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )


    # Freeze all layers first
    base_model.trainable = False


    # Fine tune last 20 layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True


    model = models.Sequential(
        [

            base_model,

            layers.GlobalAveragePooling2D(),

            layers.Dropout(0.4),

            layers.Dense(
                NUM_CLASSES,
                activation="softmax"
            )

        ]
    )


    return model
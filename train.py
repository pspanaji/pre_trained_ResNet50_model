import tensorflow as tf
from transformers import TFResNetModel
from config import IMAGE_SIZE, NUM_CLASSES, EPOCHS, MODEL_NAME


def build_and_train(train_ds, val_ds):

    # Load pretrained ResNet model
    base_model = TFResNetModel.from_pretrained(MODEL_NAME)

    # Input Layer
    input_layer = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="input_image"
    )

    # Convert NHWC -> NCHW
    x = tf.keras.layers.Lambda(
        lambda x: tf.transpose(x, [0, 3, 1, 2])
    )(input_layer)

    # Feature extraction
    x = base_model(
        pixel_values=x,
        training=False
    ).pooler_output

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # Dense Layer
    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    # Dropout
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output Layer
    output_layer = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    # Build Model
    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output_layer
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save_weights("saved_model/resnet_weights.h5")

    return model
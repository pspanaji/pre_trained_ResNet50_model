# data_loader.py
import tensorflow as tf
import numpy as np
from transformers import AutoImageProcessor
from config import IMAGE_SIZE, BATCH_SIZE, MODEL_NAME, NUM_CLASSES

# Load image processor for the ResNet model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)


def preprocess_image(image, label):
    def _preprocess(image_np, label_np):
        image_np = image_np.numpy().astype(np.uint8)
        label_np = label_np.numpy()

        # Apply Hugging Face image processing
        processed = processor(
            images=image_np,
            return_tensors="np"
        )["pixel_values"][0]

        # Convert from (3, 224, 224) --> (224, 224, 3)
        # for TensorFlow/Keras compatibility
        processed = np.transpose(processed, (1, 2, 0)).astype(np.float32)

        # Flatten label if wrapped in batch
        if label_np.ndim > 1:
            label_np = label_np.squeeze()

        return processed, label_np.astype(np.float32)

    # Wrap processing in TensorFlow
    image, label = tf.py_function(
        _preprocess,
        [image, label],
        [tf.float32, tf.float32]
    )

    # Manually set expected shapes
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    label.set_shape([NUM_CLASSES])

    return image, label


def load_datasets(train_dir, val_dir):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # Unbatch, process each image, then batch again
    train_ds = (
        train_ds
        .unbatch()
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .unbatch()
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds
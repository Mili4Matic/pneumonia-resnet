import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------------------
# Paths to dataset folders
# -------------------------------------------------------------------------
TRAIN_DIR = 'data/chest_xray/train'
VAL_DIR   = 'data/chest_xray/val'
TEST_DIR  = 'data/chest_xray/test'

# -------------------------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------------------------
IMG_SIZE           = (224, 224)
BATCH_SIZE         = 32
EPOCHS_HEAD        = 10   # For training the dense "head"(changing this number may affect on the performance of the model)
EPOCHS_FINE        = 10   # For fine-tuning the base (changing this number may affect on the performance of the model)
LEARNING_RATE      = 1e-3
LEARNING_RATE_FINE = 1e-5
UNFROZEN_LAYERS    = 30   # Number of layers to unfreeze in the base

# -------------------------------------------------------------------------
# Create data generators
# -------------------------------------------------------------------------
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_datagen   = ImageDataGenerator(rescale=1./255)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# -------------------------------------------------------------------------
# Build the model with a ResNet50 base and a custom "head"
# Also return the base_model for easy fine-tuning
# -------------------------------------------------------------------------
def build_model():
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    # Freeze all layers of ResNet
    for layer in base_model.layers:
        layer.trainable = False

    # Add our "head" on top
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=preds)
    return model, base_model

# -------------------------------------------------------------------------
# Plot training history (accuracy & loss)
# -------------------------------------------------------------------------
def plot_history(history, title_suffix=""):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title(f'Accuracy {title_suffix}')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title(f'Loss {title_suffix}')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # If you're on a server without GUI, consider using:
    # plt.savefig(f"training_curves_{title_suffix}.png")

# -------------------------------------------------------------------------
# Train only the head
# -------------------------------------------------------------------------
def train_head(model, train_gen, val_gen):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("\n[INFO] Training only the head layers...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen
    )
    return history

# -------------------------------------------------------------------------
# Fine-tune the last UNFROZEN_LAYERS of the base_model
# -------------------------------------------------------------------------
def fine_tune(model, base_model, train_gen, val_gen):
    total_layers = len(base_model.layers)
    # Unfreeze the last 'UNFROZEN_LAYERS'
    for layer in base_model.layers[total_layers - UNFROZEN_LAYERS:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n[INFO] Fine-tuning last {UNFROZEN_LAYERS} layers of ResNet50...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen
    )
    return history

# -------------------------------------------------------------------------
# Evaluate the final model on the test set
# -------------------------------------------------------------------------
def evaluate_model(model, test_gen):
    print("\n[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    # Optional environment vars to address "No algorithm worked!" issues:
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

    # Enable GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("TensorFlow version:", tf.__version__)
    print("GPUs available:", tf.config.list_physical_devices('GPU'))

    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Build model (and get base_model ref)
    model, base_model = build_model()

    # Train the head
    history_head = train_head(model, train_gen, val_gen)
    plot_history(history_head, "(Head)")

    # Fine-tune
    history_fine = fine_tune(model, base_model, train_gen, val_gen)
    plot_history(history_fine, "(Fine-tuning)")

    # Evaluate
    evaluate_model(model, test_gen)

    # Save the model
    model.save("data/models/pneumonia_resnet50.h5")
    print("\n[INFO] Model saved as pneumonia_resnet50.h5")

if __name__ == "__main__":
    main()


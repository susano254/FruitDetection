import tensorflow_datasets as tfds
import tensorflow as tf


dataset, info = tfds.load('fruits360', as_supervised=True, with_info=True)

train_dataset, validation_dataset = dataset['train'], dataset['test']
def preprocess_image(image, label):
    image = tf.image.resize(image, [150, 150])  # Resize image
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values
    return image, label

train_dataset = train_dataset.map(preprocess_image)
validation_dataset = validation_dataset.map(preprocess_image)

BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(train_dataset, epochs=20, validation_data=validation_dataset)

train_acc = history.history['accuracy']

test_loss, test_acc = model.evaluate(validation_dataset)
print('Train accuracy:', train_acc[-1])  # Print final training accuracy
print('Test accuracy:', test_acc)  # Print test accuracy

model.save('fruit_classification_cnn_model.h5')

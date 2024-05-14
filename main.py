import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image  import ImageDataGenerator, img_to_array,load_img
import matplotlib.pyplot as plt
import cv2
from glob import glob


train_path = './data/fruits-360_dataset/fruits-360/Training/'
test_path = './data/fruits-360_dataset/fruits-360/Test/'
img = load_img(train_path + "Apple Golden 1/0_100.jpg")


# plt.imshow(img)
# plt.title("Apple Golden")
# plt.axis("off")
# plt.show()



shape_of_image = img_to_array(img)
print(shape_of_image.shape)

classes = glob(train_path + "/*")
number_of_class = len(classes)
print("Number of class : " , number_of_class)

train_datagen = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip = True,
                   zoom_range = 0.3)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_path,
                                                   target_size = shape_of_image.shape[:2],
                                                   batch_size = 32,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(test_path,
                                                   target_size = shape_of_image.shape[:2],
                                                   batch_size = 32,
                                                   color_mode = 'rgb',
                                                   class_mode = 'categorical')

print(shape_of_image.shape[:2])

print(train_generator.class_indices)
print(test_generator.class_indices)
print(train_generator.image_shape)
print(test_generator.image_shape)


# num_samples = 10
# # Retrieve a batch of samples from the training generator
# for images, labels in train_generator:
#     # Create a grid for displaying images
#     plt.figure(figsize=(10, 5))
#     for i in range(num_samples):
#         # Retrieve the image and label for the current sample
#         image = images[i]
#         label = labels[i]
        
#         # Convert one-hot encoded label to class name
#         class_name = train_generator.class_indices
#         class_name = [k for k, v in class_name.items() if v == label.argmax()][0]
        
#         # Plot the image in the grid
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(image)
#         plt.title(class_name)
#         plt.axis('off')
        
#     plt.show()
#     break  # Only iterate over the first batch



# # Create a CNN model and save
# model = Sequential()
# model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
# model.add(MaxPooling2D())

# model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
# model.add(MaxPooling2D())

# model.add(Conv2D(64,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
# model.add(MaxPooling2D())

# model.add(Flatten())
# model.add(Dense(1024,activation='relu'))

# model.add(Dropout(0.5))
# model.add(Dense(number_of_class,activation = 'softmax'))

# model.compile(loss = 'categorical_crossentropy',
#               optimizer = 'rmsprop',
#               metrics = ['accuracy'])

# batch_size = 32
# number_of_batch = 1600 // batch_size
# history = model.fit(
#     train_generator,
#     steps_per_epoch = number_of_batch,
#     epochs = 100,
#     validation_data = test_generator,
#     validation_steps = 800 // batch_size)

# model.save("trial_model.h5")

# print(history.history.keys())
# plt.plot(history.history["loss"],label = "Train Loss")
# plt.plot(history.history["val_loss"],label = "Validaton Loss")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(history.history["accuracy"],label = "Train Accuracy")
# plt.plot(history.history["val_accuracy"],label = "Validaton Accuracy")
# plt.legend()
# plt.show()





model = tf.keras.models.load_model("trial_model.h5")


img = load_img(test_path + "Kiwi/3_100.jpg",target_size = shape_of_image.shape[:2])
img = img_to_array(img)
img = img / 255.0
img = img.reshape(1,*shape_of_image.shape)
result = model.predict(img)[0]

predicted_class = np.argmax(result)
print(predicted_class)
print(result[predicted_class])
print(train_generator.class_indices)
print([k for k,v in train_generator.class_indices.items() if v == result.argmax()][0])


img = load_img(test_path + "Background/0_0_0.jpg",target_size = shape_of_image.shape[:2])
img = img_to_array(img)
img = img / 255.0
img = img.reshape(1,*shape_of_image.shape)
result = model.predict(img)[0]

predicted_class = np.argmax(result)
print(predicted_class)
print(result[predicted_class])
print(train_generator.class_indices)
print([k for k,v in train_generator.class_indices.items() if v == result.argmax()][0])




# original_img = cv2.imread('data/fruits-360_dataset/fruits-360/test-multiple_fruits/cherries_wax1.jpg')
original_img = load_img('data/fruits-360_dataset/fruits-360/test-multiple_fruits/cherries_wax1.jpg')
img_array = img_to_array(original_img)
img_array = img_array / 255.0

print(img_array.shape)
# plt.imshow(original_img)
plt.axis('off')
plt.imshow(img_array)

# Define sliding window parameters
window_size = (100, 100)  # Size of the sliding window
step_size = 120  # Stride of the sliding window
classification_threshold = 0.9  # Only classify windows with a class prediction above this threshold
detections = []  # List to store the detected objects

# Initialize a list to store pyramid images
scale_factor = 0.8
pyramid_images = []

# Create the image pyramid
image = img_array
while True:
    pyramid_images.append(image)
    image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    print(image.shape)
    if image.shape[0] < 100 or image.shape[1] < 100:  # Adjust the minimum size as needed
        break

# Loop through each image in the pyramid
# Iterate over the pyramid images with the sliding window
for pyramid_level, pyramid_image in enumerate(pyramid_images):
    image_with_window = pyramid_image.copy()    
    image_with_window = cv2.cvtColor(image_with_window, cv2.COLOR_RGB2BGR)

    for y in range(0, pyramid_image.shape[0] - window_size[0] + 1, step_size):
        for x in range(0, pyramid_image.shape[1] - window_size[1] + 1, step_size):
            # Extract the current window from the image
            window = image_with_window[y:y + window_size[0], x:x + window_size[1], :]

            current_image = image_with_window.copy()
            cv2.rectangle(current_image, (x, y), (x + window_size[1], y + window_size[0]), (0, 255, 0), 2)

            current_image = cv2.resize(current_image, (800, 800))

            window = window * 255.0
            window = cv2.resize(window, (shape_of_image.shape[1], shape_of_image.shape[0])).astype('uint8')
            cv2.imshow("Window", window)
            cv2.imshow("Current Image", current_image)
            cv2.waitKey(1)

            # Preprocess the window for the model
            window = window / 255.0
            window = window.reshape(1, *shape_of_image.shape)

            # Make a prediction with the model
            result = model.predict(window)[0]
            predicted_class = np.argmax(result)
            score = result[predicted_class]
            class_name = [k for k, v in train_generator.class_indices.items() if v == predicted_class][0]
            if score > classification_threshold and class_name != 'Background':
                print(f"Detected {class_name} with score {score:.2f} at level {pyramid_level} at position ({x}, {y})")
                detections.append((x, y, window_size[1], window_size[0], predicted_class, score))



# # Display the original image with detected bounding boxes
# plt.imshow(original_img)
# plt.title("Original Image with Correct Sliding Window Detections")
# plt.axis("off")

# Plot the bounding boxes for each correct detection
for detection in detections:
    x, y, w, h, class_idx, score = detection
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.text(x, y, f"Class: {class_idx}, Score: {score:.2f}", color='r', verticalalignment='top')

plt.show()
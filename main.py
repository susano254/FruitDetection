import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image  import ImageDataGenerator, img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob


train_path = './data/fruits-360_dataset/fruits-360/Training/'
test_path = './data/fruits-360_dataset/fruits-360/Test/'
img = load_img(train_path + "Apple Golden 1/0_100.jpg")
plt.imshow(img)
plt.title("Apple Golden")
plt.axis("off")
plt.show()


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



print(train_generator.class_indices)
print(test_generator.class_indices)
print(train_generator.image_shape)
print(test_generator.image_shape)


num_samples = 10
# Retrieve a batch of samples from the training generator
for images, labels in train_generator:
    # Create a grid for displaying images
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        # Retrieve the image and label for the current sample
        image = images[i]
        label = labels[i]
        
        # Convert one-hot encoded label to class name
        class_name = train_generator.class_indices
        class_name = [k for k, v in class_name.items() if v == label.argmax()][0]
        
        # Plot the image in the grid
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis('off')
        
    plt.show()
    break  # Only iterate over the first batch

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation = 'relu', input_shape = shape_of_image.shape))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])


batch_size = 32
number_of_batch = 1600 // batch_size
history = model.fit(
    generator = train_generator,
    steps_per_epoch = number_of_batch,
    epochs = 100,
    validation_data = test_generator,
    validation_steps = 800 // batch_size)
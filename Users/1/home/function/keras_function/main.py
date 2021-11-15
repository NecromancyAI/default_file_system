import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from innocuous.MagicObj import MagicObj

def main(epochs=2, optimizer='adam', batch_size=256):
    mj = MagicObj()
    dataset_path = mj.get_dataset_path()

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(28, 28),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=(28, 28),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

    # Model Structure
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Load Checkpoint or Model
    model = mj.load_keras_model(model)

    # Train
    model.fit(train_generator, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=validation_generator,
        callbacks=[mj.callback(
                metrics={"accuracy":"accuracy"},
                filename="checkpoint.h5",
                path="/home/user/workspace/results",
                frequency=1,
                on="epoch_end")])

    # Test
    loss, accuracy = model.evaluate(validation_generator)

    mj.log(loss=loss, accuracy=accuracy)
    return loss, accuracy


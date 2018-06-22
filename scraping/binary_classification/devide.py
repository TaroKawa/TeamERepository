import os
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD


def load_dataset():
    classes = ["good", "bad"]

    width, height = 224, 224

    train_data_dir = "../../data/good_bad/train/"
    eval_data_dir = "../../data/good_bad/eval/"

    batch_size = 16
    n_epoch = 10

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.2,
        horizontal_flip=True
    )

    eval_data_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_data_gen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    eval_generator = eval_data_gen.flow_from_directory(
        eval_data_dir,
        target_size=(width, height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=16,
        shuffle=True
    )

    return train_generator, eval_generator


def get_model():
    input_tensor = Input((224, 224, 3))
    vgg = VGG16(include_top=False, input_tensor=input_tensor)

    top = Sequential()
    top.add(Flatten(input_shape=vgg.output_shape[1:]))
    top.add(Dense(256, activation="relu"))
    top.add(Dropout(0.5))
    top.add(Dense(2, activation="softmax"))

    model = Model(input=vgg.input, output=top(vgg.output))
    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(SGD(lr=1e-3, momentum=0.9),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train(model, train_gen, eval_gen):
    save_dir = "weight/"
    train_dir = "../../data/good_bad/train/"
    eval_dir = "../../data/good_bad/eval/"

    n_train = len(os.listdir(train_dir + "good/")) + \
              len(os.listdir(train_dir + "bad"))
    n_eval = len(os.listdir(eval_dir + "good")) + \
             len(os.listdir(eval_dir + "bad"))

    history = model.fit_generator(
        train_gen,
        samples_per_epoch=n_train,
        nb_epoch=10,
        validation_data=eval_gen,
        nb_val_samples=n_eval
    )

    model.save_weights(save_dir + "finetuning.h5")


if __name__ == '__main__':
    train_gen, eval_gen = load_dataset()
    model = get_model()
    train(model, train_gen, eval_gen)

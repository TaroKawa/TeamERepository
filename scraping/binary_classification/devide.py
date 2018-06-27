import os
import glob
import argparse
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint



def load_dataset():
    classes = ["good", "bad"]

    width, height = 224, 224

    train_data_dir = "../../data/good_bad/train/"
    eval_data_dir = "../../data/good_bad/eval/"

    batch_size = 128
    n_epoch = 10

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=False
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
        batch_size=32,
        shuffle=True
    )

    return train_generator, eval_generator


def get_model():
    input_tensor = Input((224, 224, 3))
    vgg = VGG16(include_top=False, input_tensor=input_tensor)

    top = Sequential()
    top.add(Flatten(input_shape=vgg.output_shape[1:]))
    top.add(Dense(1024, activation="relu"))
    top.add(Dropout(0.5))
    top.add(Dense(64, activation="relu"))
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

    callbacks = list()
    callbacks.append(ModelCheckpoint(filepath=f"weight/weight_epoch.h5"))

    history = model.fit_generator(
        train_gen,
        samples_per_epoch=n_train,
        nb_epoch=10,
        validation_data=eval_gen,
        nb_val_samples=n_eval,
        callbacks=callbacks
    )

    model.save_weights(save_dir + "finetuning_1.h5")


def classification(weight, model, dir):
    source_dir = f"{dir}/classify/"
    classify_dir = f"{dir}/good/"

    model.load_weights(weight)

    files = os.listdir(source_dir)
    preds = []
    for file in files:
        img = image.load_img(source_dir + file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(preprocess_input(x))
        preds.append(pred[0])

    for i, pred in enumerate(preds):
        if pred[0] >= pred[1]:
            os.rename(source_dir + files[i],
                      classify_dir + files[i])
        elif pred[0] < pred[1]:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train_gen, eval_gen = load_dataset()
        model = get_model()
        train(model, train_gen, eval_gen)
    elif args.mode == "classify":
        dir = "/home/arai/zipfiles/"
        model = get_model()
        weight = "weight/finetuning_1.h5"
        classification(weight, model, dir)

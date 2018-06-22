from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np


def main(path):
    model = VGG16()
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(preprocess_input(x))
    result = decode_predictions(preds, top=5)[0]
    for r in result:
        print(r)


if __name__ == '__main__':
    path = "../../data/黄色/482.jpg"
    main(path)

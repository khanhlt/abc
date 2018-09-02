from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(my_path, "dataset/ok_data")
TEST_FOLDER = os.path.join(my_path, "dataset/ng_data")


def read_images(folder):
    res = []
    img_names = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), color_mode='grayscale')
        img = img_to_array(img)
        res.append(img)
        img_names.append(filename)
    return res, img_names

def load_data():
    train, train_name = read_images(TRAIN_FOLDER)
    test, test_name = read_images(TEST_FOLDER)

    test_label = []
    for i in range(len(test)):          # 1 is anomaly, 0 is normal
        test_label.append(1)

    train = np.asarray(train)
    test = np.asarray(test)
    test_label = np.asarray(test_label)
    train_name = np.asarray(train_name)

    return train, train_name, test, test_name, test_label





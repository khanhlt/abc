from preprocess import load_data
from keras.layers import Input, Dense
from keras import Model
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K
import cv2
import os
from scipy.misc import imread

lam = 1e-4

my_path = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(my_path, "dataset/ok_data")
TEST_FOLDER = os.path.join(my_path, "dataset/ng_data")

def extract_feature(folder, vector_size=32):
    image_set = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder, filename))
        image_set.append(img)
    image_set = np.asarray(image_set)
    feature_set = []
    for image in image_set:
        try:
            alg = cv2.xfeatures2d.SURF_create()
            kps = alg.detect(image)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            kps, dsc = alg.compute(image, kps)
            dsc = dsc.flatten()
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print("Error: ", e)
            return None
        feature_set.append(dsc)
    return np.asarray(feature_set)

def contractive_loss(y_pred, y_true):
    mse = K.mean(K.square(y_true - y_pred), axis=1)
    W = K.variable(value=autoencoder.get_layer('encoded').get_weights()[0])
    W = K.transpose(W)
    h = autoencoder.get_layer('encoded').output
    dh = h * (1 - h)

    contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
    return mse + contractive

if __name__ == "__main__":
    train, train_name, test, test_name, test_label = load_data()
    train = extract_feature(TRAIN_FOLDER, 64)
    print(train.shape)
    test = extract_feature(TEST_FOLDER, 64)
    num_test_anom = len(test_label)
    train, test_norm, train_name, test_name_norm = train_test_split(train, train_name, test_size=0.1, random_state=30)
    num_test_norm = len(test_name_norm)
    test = np.concatenate((test, test_norm), axis=0)
    test_label = np.concatenate((test_label, np.zeros(len(test_norm))), axis=0)
    #
    #
    # train = train.astype('float32') / 255.
    # test = test.astype('float32') / 255.
    #
    # train = train.reshape((len(train), np.prod(train.shape[1:])))
    # test = test.reshape((len(test), np.prod(test.shape[1:])))

    nb_imgs = train.shape[0]
    print("Number images: ", nb_imgs)

    img_dim = train[0].shape[0]

    IMG_SHAPE = train[0].shape
    input_img = Input(shape=IMG_SHAPE)
    encoding_dim = 32

    encoded = Dense(128, activation='sigmoid')(input_img)
    encoded = Dense(84, activation='sigmoid')(encoded)
    encoded = Dense(encoding_dim, activation='sigmoid', name='encoded')(input_img)

    decoded = Dense(64, activation='sigmoid')(encoded)
    decoded = Dense(128, activation='sigmoid')(decoded)
    decoded = Dense(IMG_SHAPE[0], activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss=contractive_loss)
    autoencoder.fit(train, train, epochs=15, batch_size=2)

    sum, i = 0., 0.
    train_err = []
    for x in train:
        x = np.expand_dims(x, axis=0)
        loss = autoencoder.test_on_batch(x, x)
        train_err.append(loss)

    train_err = sorted(train_err, reverse=True)
    top_40 = train_err[0:int(len(train_err) * 10 / 4)]
    threshold = np.mean(top_40)

    print("Threshold: ", threshold)

    tp, fp, tn, fn = 0., 0., 0., 0.

    # loss < threshold --> normal; loss > threshold --> anomaly
    i = 0
    for x in test:
        x = np.expand_dims(x, axis=0)
        loss = autoencoder.test_on_batch(x, x)
        if (loss < threshold):  # --> normal
            print('%s: %f --> normal' % (test_label[i], loss))
            if (test_label[i] == 0):  # true negative
                tn += 1
            else:  # false negative
                fn += 1
        else:  # --> anomaly
            print('%s: %f --> anomaly' % (test_label[i], loss))
            if (test_label[i] == 0):  # false positive
                fp += 1
            else:  # true positive
                tp += 1
        i += 1

    print('\nPrecesion: %.3f' % (tp / (tp + fp)))
    print('\nRecall: %.3f' % (tp / (tp + fn)))
    print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
    print('\nAccuracy on anomaly: %d/%d' %(tp, num_test_anom))
    print('\nAccuracy on normal: %d/%d' %(tn, num_test_norm))

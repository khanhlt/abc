from preprocess import load_data
from sklearn.model_selection import train_test_split
import numpy as np
import os
from cv2 import imread
import cv2
from trim_grassmann_average.tga import TGA
import time

my_path = os.path.abspath(os.path.dirname(__file__))
TRAIN_FOLDER = os.path.join(my_path, "../dataset/ok_data")
TEST_FOLDER = os.path.join(my_path, "../dataset/ng_data")

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
            needed_size = (vector_size * 32)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print("Error: ", e)
            return None
        feature_set.append(dsc)
    return np.asarray(feature_set)


if __name__=="__main__":
    train, train_name, test, test_name, test_label = load_data()
    train = extract_feature(TRAIN_FOLDER)
    print(train.shape)
    test = extract_feature(TEST_FOLDER)

    num_test_anom = len(test_label)
    train, test_norm, train_name, test_name_norm = train_test_split(train, train_name, test_size=0.2, random_state=30)
    num_test_norm = len(test_name_norm)
    test = np.concatenate((test, test_norm), axis=0)
    test_label = np.concatenate((test_label, np.zeros(len(test_norm))), axis=0)

    start = time.time()

    tga = TGA(n_components=10, random_state=1)
    tga.fit(train)
    center = tga.center_
    print("center: ", tga.center_)

    train_dist = []
    for x in train:
        d = np.mean(((x - center) ** 2))
        train_dist.append(d)
    train_dist = sorted(train_dist, reverse=True)
    top_50 = train_dist[0:int(len(train_dist) / 2)]
    threshold = np.mean(top_50)
    print("threshold: ", threshold)
    print("mean: ", np.mean(train_dist))
    print("mean + std: ", np.mean(train_dist) + np.std(train_dist))

    threshold = np.mean(train_dist) + np.std(train_dist)

    end = time.time()

    k = 0
    for x in test:
        print("distance: ", np.mean(((x - center) ** 2)))
        k += 1
        if (k == len(test) - len(test_norm)):
            print("cut here")

    tp, fp, tn, fn = 0., 0., 0., 0.
    for i in range(0, len(test) - len(test_norm)):
        d = np.mean(((test[i] - center) ** 2))
        if (d > threshold):
            tp += 1
        else:
            fn += 1
    for i in range(len(test) - len(test_norm), len(test)):
        d = np.mean(((test[i] - center) ** 2))
        if (d > threshold):
            fp += 1
        else:
            tn += 1


    print('\nPrecesion: %.3f' % (tp / (tp + fp)))
    print('\nRecall: %.3f' % (tp / (tp + fn)))
    print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
    print('\nAccuracy on anomaly: %d/%d' % (tp, num_test_anom))
    print('\nAccuracy on normal: %d/%d' % (tn, num_test_norm))
    print('\nLearning time: ', end - start)



from preprocess import load_data
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
from scipy.misc import imread
from sklearn import svm
from sklearn.model_selection import ParameterGrid

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

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(train)
    pred_test = clf.predict(test)
    print(pred_test)


    tp, fp, tn, fn = 0., 0., 0., 0.
    for i in range(0, len(test) - len(test_norm)):
        if (pred_test[i] == -1):
            tp += 1
        elif (pred_test[i] == 1):
            fn += 1
    for i in range(len(test) - len(test_norm), len(test)):
        if (pred_test[i] == -1):
            fp += 1
        elif (pred_test[i] == 1):
            tn += 1

    print('\nPrecesion: %.3f' % (tp / (tp + fp)))
    print('\nRecall: %.3f' % (tp / (tp + fn)))
    print('\nAccuracy: %.3f' % ((tp + tn) / (tp + tn + fn + fp)))
    print('\nAccuracy on anomaly: %d/%d' % (tp, num_test_anom))
    print('\nAccuracy on normal: %d/%d' % (tn, num_test_norm))
from preprocess import load_data
import cv2
import numpy as np

if __name__=="__main__":
    train, train_name, test, test_name, test_label = load_data()
    train_kaze = []
    for t in train:
        alg = cv2.KAZE_create()
        kps = alg.detect(t)
        kps = sorted(kps, key=lambda x: -x.response)[:32]
        kps, dsc = alg.compute(t, kps)
        dsc = dsc.flatten()
        train_kaze.append(dsc)
    train_kaze = np.asarray(train_kaze)
    print(train_kaze.shape)

from preprocess import load_data
from keras.layers import Input, Dense
from keras import Model
import numpy as np
import keras.backend as K

LAMBDA = 0.1


def discriminative_labeling(rec_err_origin, train):
    rec_err = sorted(rec_err_origin, reverse=True)
    l = len(rec_err)
    print("check rec_err len: ", l)
    thres = np.array([rec_err[int(l * 5 / 100)], rec_err[int(l * 10 / 100)], rec_err[int(l * 15 / 100)],
                      rec_err[int(l * 20 / 100)], rec_err[int(l * 25 / 100)], rec_err[int(l * 30 / 100)]])
    print("check thres: ", thres)
    h_min = 10e4
    thr = 0
    for t in thres:
        norm_err = [x for x in rec_err if x < t]
        anom_err = [x for x in rec_err if x >= t]

        a = (np.array(norm_err) - np.array([np.mean(norm_err)] * len(norm_err))) ** 2
        b = (np.array(anom_err) - np.array([np.mean(anom_err)] * len(anom_err))) ** 2
        c = (np.array(rec_err) - np.array([np.mean(rec_err)] * len(rec_err))) ** 2
        sigma_w = np.sum(a) + np.sum(b)
        sigma_t = np.sum(c)
        h = sigma_w / sigma_t
        if h < h_min:
            h_min = h
            thr = t
    norm, anom, norm_index = [], [], []
    for i in range(len(rec_err_origin)):
        if (rec_err_origin[i] < thr):
            norm.append(train[i])
            norm_index.append(i)
        else:
            anom.append(train[i])

    return np.array(norm), np.array(anom), h_min, np.array(norm_index)


def loss_func(y_true, y_pred, norm_index, h):
    norm_pred = [y_pred[i] for i in norm_index]
    norm_true = [y_true[i] for i in norm_index]
    norm_pred = np.asarray(norm_pred)
    norm_true = np.asarray(norm_true)
    mse = K.mean(K.square(norm_true - norm_pred), axis=1)
    extra = LAMBDA * h
    return mse + extra

def loss(norm_index, h):
    def dice(y_true, y_pred):
        return loss_func(y_true, y_pred, norm_index, h)
    return dice


if __name__ == "__main__":
    train, train_name, test, test_name, test_label = load_data()
    train = train.astype('float32') / 255.
    test = test.astype('float32') / 255.

    train = train.reshape((len(train), np.prod(train.shape[1:])))
    test = test.reshape((len(test), np.prod(test.shape[1:])))

    IMG_SHAPE = train[0].shape
    input_img = Input(shape=IMG_SHAPE)
    encoding_dim = 32
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(IMG_SHAPE[0], activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    train = np.asarray(train)

    autoencoder.fit(train, train, epochs=1, batch_size=2)

    rec_err = []
    for x in train:
        x = np.expand_dims(x, axis=0)
        err = autoencoder.test_on_batch(x, x)
        rec_err.append(err)
    print(len(rec_err))
    norm, anom, h, norm_index = discriminative_labeling(np.array(rec_err), np.array(train))

    anom_name = []

    for i in range(len(train)):
        if train[i] in anom:
            anom_name.append(train_name[i])

    print(norm.shape)
    print(anom.shape)
    print(anom_name)

    norm_pred, norm_true = [], []
    # get the loss function
    model_loss = loss(norm_index=norm_index, h=h)
    autoencoder.compile(optimizer='adam', loss=model_loss)
    autoencoder.fit(train, train, epochs=1, batch_size=2)

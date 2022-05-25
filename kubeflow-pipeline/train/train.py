from argparse import ArgumentParser

from numpy import concatenate
from numpy.random import seed
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow import random as tfrandom
from pandas import read_csv

def split_ts(ts, n):
    # split into train and test sets
    values = ts.values
    n_train_hours = len(ts)-n
    train = values[:n_train_hours, :]
    print('Data Train: ', len(train))
    val = values[n_train_hours:, :]
    print('Data Validation: ', len(val))
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    val_X, val_y = val[:, :-1], val[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)
    return train_X, train_y, val_X, val_y

def set_testset(ts):
    test = ts.values
    print('Data Test: ', len(test))
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(test_X.shape, test_y.shape)
    return test_X, test_y


def train(args):
    patient_id = args.patient
    path = "Datasets/%s/" % (patient_id)
    filename_train = '%s-training(t+30)' % (patient_id)
    datafile_train = path + 'patient_' + filename_train + '.csv'
    ts = read_csv(datafile_train)

    # seeds for random generators
    seeds = [42]
    repeats = 1
    #error_scores = list()
    for r in range(repeats):
        # inizialize random generators
        seed(seeds[r])
        tfrandom.set_seed(seeds[r])

        # set datasets
        train_X, train_y, val_X, val_y = split_ts(ts, 1000)   

        # design network
        lstm_model = keras.Sequential()
        lstm_model.add(keras.layers.LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
        lstm_model.add(keras.layers.Dense(1))
        lstm_model.compile(loss='mae', optimizer='adam')

        #fit network
        history = lstm_model.fit(train_X, train_y, epochs=10, batch_size=10, validation_data=(val_X, val_y), verbose=2, shuffle=False)
        lstm_model.save('models/' + args.model + '.hdf5')
 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--patient', type=str, default='540')
    parser.add_argument('--model', type=str,default='model')
    args = parser.parse_args()

    train(args)
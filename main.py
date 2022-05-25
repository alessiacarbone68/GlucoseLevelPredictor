import random

from numpy import sqrt
from numpy import concatenate
from numpy.random import seed
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime
from ErrorGrid import clarke_error_grid
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow import random as tfrandom


# convert time series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def plot_data(values, dataset, plot_show, filename):
    # specify columns to plot
    groups = [0, 1, 2]
    i = 1
    # plot each column
    pyplot.figure(figsize=(16, 9))
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        if group == 0:
            pyplot.plot(values[:, group], alpha=0.5, lw=1, label='$G(t)$')
        else:
            if group == 1:
                pyplot.plot(values[:, group], alpha=0.5, lw=1, label='I(t)')
            else:
                pyplot.plot(values[:, group], alpha=0.5, lw=1, label='Cho(t)')
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.xlabel('Time')
    pyplot.savefig(filename, dpi=150)
    if (plot_show == 1):
        pyplot.show()


def load_data(file_in, file_out, plot_show, plotfile_data):
    dataset = read_csv(file_in)
    # manually specify column names
    dataset.columns = ['glucose', 'insulin', 'cho']
    # save data rearranged to file
    dataset.to_csv(file_out)

    # load training set
    dataset = read_csv(file_out, header=0, index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    #values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    time_series = values.astype('float32')

    # Plot data
    # plot_data(time_series, dataset, plot_show, plotfile_data)

    scaled_ts = time_series
    # frame as supervised learning
    # from t-30 to t+30
    reframed_ts = series_to_supervised(scaled_ts, 3, 4)
    # from t-60 to t+30
    #reframed_ts = series_to_supervised(scaled_ts, 14, 7)
    # drop columns we don't want to predict
    # from t-30 to t+30
    reframed_ts.drop(reframed_ts.columns[[12,13,14,15,16,17,19,20]],
                     axis=1, inplace=True)
    # from t-60 to t+30
    #reframed_ts.drop(reframed_ts.columns[[45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62]],
    #                 axis=1, inplace=True)
    return reframed_ts

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


def main():
    path = "Datasets/540/"
    filename_train = '540-training(t+30)'
    filename_test = '540-testing(t+30)'
    file_train_in = path + filename_train + '.csv'
    datafile_train = path + 'patient_' + filename_train
    plotfile_data_train = path + 'dataplot_' + filename_train + '.png'
    file_test_in = path + filename_test + '.csv'
    datafile_test = path + 'patient_' + filename_test
    plotfile_data_test = path + 'dataplot_' + filename_test + '.png'
    plot_show = 1

    ts = load_data(file_train_in, datafile_train, plot_show, plotfile_data_train)
    ts_test = load_data(file_test_in, datafile_test, plot_show, plotfile_data_test)

    # seeds for random generators
    seeds = [42]
    repeats = 1
    error_scores = list()
    for r in range(repeats):
        # inizialize random generators
        seed(seeds[r])
        tfrandom.set_seed(seeds[r])
        # set datasets
        train_X, train_y, val_X, val_y = split_ts(ts, 1000)
        test_X, test_y = set_testset(ts_test)
        # design network
        lstm_model = keras.Sequential()
        lstm_model.add(keras.layers.LSTM(200, input_shape=(train_X.shape[1], train_X.shape[2])))
        lstm_model.add(keras.layers.Dense(1))
        lstm_model.compile(loss='mae', optimizer='adam')
        #fit network
        history = lstm_model.fit(train_X, train_y, epochs=10, batch_size=10, validation_data=(val_X, val_y), verbose=2, shuffle=False)
        #plot history
        #pyplot.plot(history.history['loss'], label='train')
        #pyplot.plot(history.history['val_loss'], label='test')
        #pyplot.legend()
        #pyplot.show()

        lstm_model.save('models/model.hdf5')

        # make a prediction
        yhat = lstm_model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        #inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        #inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        # Compute RMSE on the TEST Set
        mae = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test Set MAE: ', mae)
        error_scores.append(mae)

        # plot
        #pyplot.plot(inv_y, label='Original')
        #pyplot.plot(inv_yhat, label='Fitted')
        #pyplot.show()

        #clarke_fig, zone = clarke_error_grid(inv_y*18, inv_yhat*18, 'Clarke Error Grid')
        #print("Clarke Error Grid Zones")
        #print(zone)
        #pyplot.show()

    # summarize results
    #results = DataFrame()
    #results['mae'] = error_scores
    #print(results.describe())
    #results.boxplot()
    #pyplot.show()


if __name__ == '__main__':
    main()

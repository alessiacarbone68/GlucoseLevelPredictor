from argparse import ArgumentParser
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import datetime

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

def load_data(file_in):
    dataset = read_csv(file_in)
    dataset.columns = ['glucose', 'insulin', 'cho']

    values = dataset.values
    time_series = values.astype('float32')
    scaled_ts = time_series
    reframed_ts = series_to_supervised(scaled_ts, 3, 4)
    reframed_ts.drop(reframed_ts.columns[[12,13,14,15,16,17,19,20]],
                     axis=1, inplace=True)
                     
    return reframed_ts

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--patient', type=str, default='540')    
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    patient_id = args.patient
    path = "Datasets/%s/" % (patient_id)
    
    filename_train = '%s-training(t+30)' % (patient_id)
    filename_test = '%s-testing(t+30)' % (patient_id)
    
    file_train_in = path + filename_train + '.csv'
    datafile_train = path + 'patient_' + filename_train + '.csv'

    file_test_in = path + filename_test + '.csv'
    datafile_test = path + 'patient_' + filename_test + '.csv'

    train_data = load_data(file_train_in)
    test_data = load_data(file_test_in)

    train_data.to_csv(datafile_train)
    test_data.to_csv(datafile_test)






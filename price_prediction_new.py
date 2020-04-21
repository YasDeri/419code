import numpy as np
np.random.seed(123)
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
import pandas as pd
from tqdm import tqdm
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

TIME_STEPS = 3
BATCH_SIZE = 2
lr = 0.001
your_epochs=20

# reset states after every epoch is complete - to be used for stateful LSTM
class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()

# time window size or the sequence length for each sample is defined by time steps
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in tqdm(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]

    print("length of time-series i/o", x.shape, y.shape)
    return x, y

# trimming dataset is needed for stateful = True LSTM which is needed for our use case
def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

# scale each column and store min max scalar for use while predicting using the trained NN model
def scale_data(train, test, label_col):
    scalar = MinMaxScaler()
    label_scalar = None
    for col in train.columns:
        train[col] = scalar.fit_transform(train[col].values.reshape(-1,1))
        test[col] = scalar.transform(test[col].values.reshape(-1,1))
        if col == label_col:
            label_scalar = scalar

    return train, test, label_scalar


# load csv file
df_ge = pd.read_csv("sp-index-final.csv")

# create lagged variables in dataframe using Time steps

# drop ununsed variables
print(df_ge.columns)
cols_to_use = ['Close', '12_Day_EMA', '26_Day_EMA', 'RSI', 'MACD_Signal', 'MACD', 'Volume', 'MACD_Histogram']


# create a copy not to effect rest of the code
data = df_ge[cols_to_use].copy()

# predicting next time step closing value
data['target'] = data['Close'].shift(-1)

# change object types to float and create lagged values immitating sequences used for LSTM
for col in cols_to_use:
    if col == 'Volatility':
        data[col] = data[col].str.replace(r'%', r'0').astype('float') / 100.0
    if data[col].dtype == object:
        data[col] = data[col].astype(float)
    for time_step in range(1, TIME_STEPS):
        name = col + '_lag_' + str(time_step)
        data[name] = data[col].shift(time_step)

# print(data.shape)


# remove nan values
data.dropna(inplace=True)
data.to_csv("check_data.csv", index=False)

# print(data.shape)
# print(data.head())

# split into training and test set
df_train, df_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False, random_state=123)

# shuffle training set
df_train = df_train.sample(frac=1)

# create training and testing sets
x_train = df_train.loc[:, df_train.columns != 'target']
y_train = df_train.loc[:, 'target']
x_test = df_test.loc[:, df_test.columns != 'target']
y_test = df_test.loc[:, 'target']

print("Training set shape: ", x_train.shape)
print("Test set shape: ", x_test.shape)
print(x_train.dtypes)

# fit xgboost
model = xgb.XGBRegressor()
model.fit(x_train, y_train)

# create a dataframe to store feature importances
feat_imp = pd.DataFrame()
feat_imp["features"] = x_train.columns
feat_imp["importances"] = model.feature_importances_/sum(model.feature_importances_)

# sort in descending order
feat_imp.sort_values(by="importances", ascending=False, inplace=True)

# plot feature importances
plt.bar(feat_imp["features"], feat_imp["importances"])
plt.xticks(rotation='vertical')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importances Plot")
plt.show()

# mse score without optimization
y_pred = model.predict(x_test)
print("Mean Squared Error with no optimization: ", mean_squared_error(y_test, y_pred))

thresholds = np.sort(model.feature_importances_)

for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(x_train)
	# train model
	selection_model = xgb.XGBRegressor()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(x_test)
	y_pred = selection_model.predict(select_X_test)
	mse = mean_squared_error(y_test, y_pred)
	print("Thresh=%.3f, n=%d, mse: %.2f" % (thresh, select_X_train.shape[1], mse))

print(df_ge.tail())

# plt.figure()
# plt.plot(df_ge["Close"])
# plt.plot(df_ge["Gold_Close"])
# plt.title('Daily S&P 500 Index Price History')
# plt.ylabel('Price (USD)')
# plt.xlabel('Days')
# plt.legend(['Close'], loc='upper left')
# plt.show()

df_train, df_test = train_test_split(df_ge, train_size=0.8, test_size=0.2, shuffle=False, random_state=123)

# plt.figure()
# plt.plot(df_ge["Volume"])
# plt.title('Daily S&P 500 Traded Volume History')
# plt.ylabel('Volume')
# plt.xlabel('Days')
# plt.show()

plt.figure()
plt.plot(df_ge["Close"])
plt.title('Daily Close Price History')
plt.ylabel('Close')
plt.xlabel('Days')
plt.show()

print("checking if any null values are present\n", df_ge.isna().sum())

# see relevant columns for our task
print(df_ge.columns)

# train_cols = ["Close","Volume"]

# columns used for training
train_cols = ['26_Day_EMA', 'Close', '12_Day_EMA']

# train test split - ensure random state is entered for repeatability - each subsequent run will use the same train test split
df_train, df_test = train_test_split(df_ge.loc[:,train_cols], train_size=0.8, test_size=0.2, shuffle=False, random_state=123)
print("Train and Test size", len(df_train), len(df_test))

#inverse transformation is required to change predictions to actual values scale hence MinMaxScalar should be saved
# x = df_train.loc[:,train_cols].values
# min_max_scaler = MinMaxScaler()
# print(x[0])
# x_train = min_max_scaler.fit_transform(x)
# x_test = min_max_scaler.transform(df_test.loc[:,train_cols])


# scale columns and store min max scalar value
x_train, x_test, scalar = scale_data(df_train, df_test, "Close")

#In the earlier version label is hard coded to column number 1 - now it is the last column in this list
#train_cols = ['Open', 'High', 'Low', 'RSI', 'MACD_Signal', 'Gold_Close', "Close"]

# x_t, y_t = build_timeseries(x_train, 1)
# x_t = trim_dataset(x_t, BATCH_SIZE)
# y_t = trim_dataset(y_t, BATCH_SIZE)
# x_temp, y_temp = build_timeseries(x_test, 1)

# build time series. Last columns ie Close column is used for
x_t, y_t = build_timeseries(x_train.values, -1)
x_temp, y_temp = build_timeseries(x_test.values, -1)

# shuffle training dataset (time series is now present in the form of each individual sequence)
x_t, y_t = shuffle(x_t, y_t)

# build model function for keras wrapper
def build_model(neurons, opt, num_layers):
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    lstm_model.add(Dropout(0.4))
    for i in range(num_layers):
      lstm_model.add(Dense(neurons,activation='relu'))
    lstm_model.add(Dense(1,activation='linear'))
    optimizer = opt(lr=lr)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return lstm_model

# sklearn wrapper for keras
model = KerasRegressor(build_fn=build_model, verbose=2, epochs=your_epochs)

# define parameter grid
batch_size = [2,20]
neurons = [20, 60]
num_layers = [2]
opt = [optimizers.Adam, optimizers.Adagrad]


# dictionary to create parameter grid
param_grid = dict(batch_size=batch_size, neurons=neurons, opt = opt, num_layers = num_layers)

# define gridsearchcv model
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=5,
                    return_train_score=True)

# fit function to initiate optimization
grid_result = grid.fit(x_t, y_t)

# define dataframe to store results
results = pd.DataFrame(grid_result.cv_results_)

relevant_cols = ['param_batch_size', 'param_neurons',	'param_num_layers',	'param_opt',
                 'mean_test_score',	'std_test_score',	'mean_train_score',	'std_train_score']

# save results to csv file
results[relevant_cols].to_csv("stored_results_.csv", index=False)

# test parameters
results = pd.read_csv("stored_results.csv", sep=";")

# find maximum values of test MSE for each batch size
plotting = results.groupby('param_batch_size')['mean_test_score'].max().reset_index()
plt.plot(plotting['param_batch_size'], abs(plotting['mean_test_score']))
plt.xticks(plotting['param_batch_size'].values)
plt.xlabel("Batch size")
plt.ylabel("MSE (Test Set)")
plt.title(" MSE for diff Batch Sizes")
plt.show()

print(results.columns)

# gridsearch cv maximizes the -ve mse (same as minimising mse)
results = results.sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)
print(results['mean_test_score'].head())

# store best parameters in variables
batch_size = results.loc[0,'param_batch_size']
neurons = results.loc[0,'param_neurons']

# gridsearch stores optimizer as string hence this method had to be used to make it usable directly
opt = optimizers.Adagrad if 'Adagrad' in results.loc[0,'param_opt'] else optimizers.Adam
num_layers = results.loc[0,'param_num_layers']

print("Best Parameters: ", batch_size, neurons, opt, num_layers)











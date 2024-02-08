from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from preprocessing import *  
from sklearn.svm import SVR
import numpy as np
import itertools
from joblib import Parallel, delayed
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.neighbors import KNeighborsRegressor




def train_knn(x_train, y_train, x_val, y_val):


    def objective(trial):
    
        n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
                
        
        
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        knn.fit(x_train, y_train)
        
        
        predictions = knn.predict(x_val)
        mse = MSE(y_val, predictions)
    
        return mse
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30) 

    
    best_params = study.best_params
    best_mse = study.best_value
    model = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
    model.fit(x_train, y_train)

    return model, best_mse



def train_lstm(x_train, y_train, x_val, y_val):	
    print("Train LSTM")
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    tf.keras.backend.clear_session()

    def create_model(params):

        units = params['units']
        learning_rate = params['learning_rate']
        activation = params['activation']
        optimizer_name = params['optimizer']

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)

        model = Sequential()
        model.add(LSTM(units=units, activation=activation, input_shape=(1, x_train.shape[1])))
        model.add(Dense(1))

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)

        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model



    def objective(trial):
        params = {}
        params['units'] = trial.suggest_int('units', 1, 100)
        params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log = True)
        params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
        params['optimizer'] = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])


        model = create_model(params)
        
        mse_execs = []
        
        for i in range(0, 2):
            
            model.fit(trainX, y_train, epochs=20, batch_size=1, verbose=0)
            prev_v = model.predict(valX)
            mse = MSE(y_val, prev_v)
            mse_execs.append(mse)

        return np.mean(mse_execs)



    trainX = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    valX = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42) )
    study.optimize(objective, n_trials=30, n_jobs=-1)  
    best_mse = study.best_value
    
    best_params = study.best_params

    model = create_model(best_params)
    model.fit(trainX, y_train, epochs=20, batch_size=1, verbose=0)
    
    return model, best_mse





def train_svr(x_train, y_train, x_val, y_val):
    print("Train SVR")
 
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'sigmoid'])
        gamma = trial.suggest_float('gamma', 1e-5, 1e3, log=True)
        epsilon = trial.suggest_float('epsilon', 1e-5, 1e-1, log=True)
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)

        modelo = SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=C)
        modelo.fit(x_train, y_train)
        prev_v = np.absolute(modelo.predict(x_val))
        
        mse = MSE(y_val, prev_v)

        return mse
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42) )
    study.optimize(objective, n_trials=100, n_jobs=-1)  
    best_mse = study.best_value
    
    best_params = study.best_params
    best_kernel = best_params['kernel']
    best_gamma = best_params['gamma']
    best_epsilon = best_params['epsilon']
    best_C = best_params['C']
    melhor_modelo = SVR(kernel=best_kernel, gamma=best_gamma, epsilon=best_epsilon, C=best_C)
    return melhor_modelo, best_mse





def train_linear_regression(x_train, y_train, x_val, y_val):
    print("Train LR")
   
    modelo = LinearRegression()
    modelo.fit(x_train, y_train)
    prev_v = modelo.predict(x_val) 
    melhor_mse  = MSE(y_val, prev_v)
    
    return modelo, melhor_mse 
import numpy as np
np.random.seed(42)
from sklearn.metrics import mean_squared_error as MSE
from preprocessing import *  
from modelos import *
import pandas as pd
import pickle

import os
from joblib import dump



def reamostragem(serie, n):

    size = len(serie)
    
    ind_particao = []
    for i in range(n):
        ind_r = np.random.randint(size) 
        ind_particao.append(ind_r)
        
    
    return ind_particao


def fit_model(model, X_train, y_train, X_val, y_val):
    
    
    
    
    model_mapping = {
        'LinearRegression' : train_linear_regression,
        'SVR' : train_svr,
        'LSTM' : train_lstm,
        'KNN' : train_knn,
    }
    
    model, _ = model_mapping[model](X_train, y_train, X_val, y_val)
    if model:
        return model
    else:
        print("Modelo não encontrado!") 


def save_model(model, serie_name, model_name, i):
    path_folder = serie_name+'/'+model_name
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    
    path_to_save = os.path.join(path_folder, f"model_{i}.joblib")
    dump(model, path_to_save)


def bagging(model_name, qtd_modelos, X_train, y_train, serie_name):
    
    ens = []
    ind_train = []
    ind_particao = []
    ensemble = {'models':[], 'indices': [], 'indices_train': [] } 


    if len(y_train.shape) == 1:
        y_train =  y_train.reshape(len(y_train), 1)
    
    
    train = np.hstack([X_train, y_train])
    
    for i in range(0, qtd_modelos):

        print('Training model: ', i)

        tam = len(train)
        
        indices = reamostragem(train, tam)

        particao = train[indices, :] 
        
        Xtrain, Ytrain = particao[:, 0:-1], particao[:, -1] 
        tam_val = int(len(Ytrain)*0.32)  
        x_train = Xtrain[0:-tam_val, :]
        y_train = Ytrain[0:-tam_val]
        x_val = Xtrain[-tam_val:, :]
        y_val = Ytrain[-tam_val:]

        
        indices_train = indices[:-int(len(Ytrain)*0.32)]
        
        
        model = fit_model(model_name, x_train, y_train, x_val, y_val) 
        save_model(model, serie_name, model_name, i)
        # ens.append(model)
        ind_train.append(indices_train)
        ind_particao.append(indices)
        
    

    # ensemble['models'] = ens
    # ensemble['indices'] = ind_particao
    # ensemble['indices_train'] = ind_train #indices utilizados para realizar o treinamento dos modelos
    
    return ensemble



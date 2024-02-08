
import numpy as np
#from typing import *
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf



class Preproc:

    def __init__(self, ):
        self.scaler = None 
        self.samples = None

    def normalise_interval(self, train_sample: Series, minimo: float = 0.0, maximo: float = 1.0):
        if self.scaler:
            normalized = self.scaler.transform(train_sample.reshape(-1, 1))
            return normalized
        
        else:

            scaler = MinMaxScaler(feature_range=(minimo, maximo))
            self.scaler = scaler.fit(train_sample.reshape(-1, 1))
            normalized = scaler.transform(train_sample.reshape(-1, 1))
            return normalized 
	
		
    def desnorm_interval(self, serie_norm):        
        inversed = self.scaler.inverse_transform(serie_norm)
        return inversed

    def split_serie(self, serie, perc_train = 0.6, perc_val = 0.0):

        dict_samples = {}

        train_size = int(round(len(serie) * perc_train))


        if perc_val > 0:
            val_size = int(round(len(serie) * perc_val))
            dict_samples['train'] = serie[0:train_size]
            dict_samples['val'] = serie[train_size:train_size+val_size]
            dict_samples['test'] = serie[train_size+val_size:]

            
                   
        else:
            
            dict_samples['train'] = serie[0:train_size]
            dict_samples['test'] = serie[train_size:]

        self.samples = dict_samples
        return dict_samples
  

    def create_windows(self, serie, window_size):
        window_size += 1 #add the target
        list_of_sliding_windows = []
        
        for i in range(window_size, len(serie)+1):
            window = serie[0:i][-window_size:].flatten()
            list_of_sliding_windows.append(window)

        X_y = np.hstack(list_of_sliding_windows).reshape(len(list_of_sliding_windows), window_size)
        X, y = X_y[:, 0: -1], X_y[:, -1]
        X = X[:, ::-1] #to sequence -> [lag1, lag2, ..., lagM]
        return X, y
    
    def select_lag_acf(self, serie, max_lag):  #pacf
         #A função acf que você mencionou é parte do módulo statsmodels.tsa.stattools da biblioteca Statsmodels em Python e é usada para calcular a função de autocorrelação (ACF) de uma série temporal. A função de autocorrelação é uma medida estatística que indica o grau de correlação entre uma série temporal e suas próprias versões passadas em vários lags (atrasos temporais).
       
        acf_x, confint = acf(serie, nlags=max_lag, alpha=.05) #A função acf é útil para entender a autocorrelação na série temporal, o que pode ser importante em análise de séries temporais, modelagem e previsão. Através da ACF, você pode identificar padrões de sazonalidade e determinar a ordem de modelos autorregressivos (AR) em processos autorregressivos integrados de média móvel (ARIMA).
        
        
        limiar_superior = confint[:, 1] - acf_x 
        limiar_inferior = confint[:, 0] - acf_x

        lags_selecionados = []
        
        for i in range(1, max_lag+1):

            
            if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
                lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0
        
        #caso nenhum lag seja selecionado, essa atividade de seleção para o gridsearch encontrar a melhor combinação de lags
        if len(lags_selecionados)==0:


            print('NENHUM LAG POR ACF')
            lags_selecionados = [i for i in range(max_lag)]

        print('LAGS', lags_selecionados)
        #inverte o valor dos lags para usar na lista de dados
        # lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]



        return lags_selecionados

		
    def select_validation_sample(self, serie, perc_val):
        tam = len(serie)
        val_size = np.fix(tam *perc_val).astype(int)
        return serie[0:tam-val_size,:],  serie[tam-val_size:-1,:]





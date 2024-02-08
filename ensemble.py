import numpy as np
from bagging import bagging
import os
from joblib import dump



class Ensemble:

    def __init__(self, size_pool, models_name, serie_name, approach_generate = 'bagging'):
        self.historic = None 
        self.combine_approch = 'mean'
        self.models_name = models_name
        self.approach = approach_generate
        self.pool = []
        self.serie_name = serie_name
        self.size_pool = size_pool
        self.iter_model  = 0


    def fit(self, X_train, y_train):

        if self.approach == 'bagging':
            pool = bagging(self.models_name, self.size_pool, X_train, y_train, self.serie_name)
            self.pool = pool

        
        else:
            print("Abordagem não implementada!")





    def save_models(self, serie_name, model_name):
        path_folder = serie_name+'/'+model_name
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

        for i, model in enumerate(self.pool['models']):
            path_to_save = os.path.join(path_folder, f"model_{i}.joblib")
            dump(model, path_to_save)
        


    def predict(self, X, comb = 'mean', size_pool = None):

        y_hats = []

        if len(X.shape) == 1:
            X = X.reshape(1, -1) 

        for x_input in X:

            prevs = []
            if size_pool:
                #pruning pool
                for model in self.pool['models'][0:size_pool]:
                    y = model.predict(x_input.reshape(1, -1))[0]
                    prevs.append(y)
            
            else:

                for model in self.pool['models']:
                    y = model.predict(x_input.reshape(1, -1))[0]
                    prevs.append(y)

        
            if comb == 'mean':
                y_hats.append( np.mean(prevs) )

            elif comb == 'median':
                y_hats.append( np.median(prevs) )

            else:
                return None
        
        return y_hats


        





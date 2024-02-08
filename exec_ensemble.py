from preprocessing import Preproc
import pandas as pd
import numpy as np

import time


series = ['airline']
for serie in series:
    path_serie = f'/home/eraylson/union/timeseries/{serie}.txt'
    df_serie = pd.read_csv(path_serie, header=None)
    pre = Preproc()
    serie_np = df_serie.to_numpy().reshape(-1, )
    dict_samples = pre.split_serie(serie_np, perc_train = 0.75)
    train, test = dict_samples['train'], dict_samples['test']
    _ = pre.normalise_interval(train)
    train_norm = pre.normalise_interval(train)
    test_norm = pre.normalise_interval(test)
    idx_lags = pre.select_lag_acf( train_norm, 20)
    X_train, y_train = pre.create_windows(train_norm, np.max(idx_lags)+1)

    qtd_lags = np.max(idx_lags)
    test_with_previous = np.vstack([train_norm[-qtd_lags-1:], test_norm])
    X_test, y_test = pre.create_windows(test_with_previous, np.max(idx_lags)+1)
    
    X_train = X_train[:, idx_lags]
    X_test = X_test[:, idx_lags]

    from ensemble import Ensemble
    ens = Ensemble(100, 'LSTM', approach_generate='bagging', serie_name = serie)
    start = time.time()
    
    ens.fit(X_train, y_train)
    end = time.time()
    duration = end - start
    #ens.save_models(serie, 'lstm')
    with open(f'{serie}_pool_knn_time.txt', 'w') as f:
        f.write(f"tempo de execução: {duration}")
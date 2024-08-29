import numpy as np
import pandas as pd
import os
import math

def get_data(path_csv):
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        data = pd.read_csv(path_csv, delimiter=';')
        fitur = data.iloc[:, :-1].values # Fitur iris
        target = data.iloc[:, -1].values # Target kelas (kolom terakhir)
        return fitur,target
    except:
        print('file tidak bisa dibaca')

def error(target, prediksi):
    selisih=target-prediksi
    return math.pow(selisih,2)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def aktivasi(x):
    return np.where(x >= 0.5, 1, 0)

def akurasi(target, prediksi):
    return np.mean(target == prediksi)*100
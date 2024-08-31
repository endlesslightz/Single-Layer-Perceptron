import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
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

def cetak_graf(m_mse, m_acc):
       # cetak chart error di subplot 1
    x = np.arange(1,11) 
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x,m_mse, marker = 'o')
    axs[0].set_title('Grafik MSE') 
    axs[0].set_ylabel('Error') 
    axs[0].set_xlabel('Epoch ke-')
    # cetak chart akurasi di subplot 2
    axs[1].plot(x,m_acc, marker = 'o')
    axs[1].set_title('Grafik Akurasi') 
    axs[1].set_ylabel('Akurasi (%)') 
    axs[1].set_xlabel('Epoch ke-')
    plt.show(block=True)
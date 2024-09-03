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

def loss(target, prediksi):
    # hitung MSE tiap data per epoch
    sum_err=0
    for i in range(target.size):
        sum_err +=error(target[i], sigmoid(prediksi[i]))
    sum_err=sum_err/target.size
    return sum_err

def cetak_graf(m_err_latih, m_err_uji, m_aku_latih, m_aku_uji):
    # cetak chart error di subplot 1
    x = np.arange(0,11) 
    # di-concatenate dulu pakai nilai sebelum pelatihan dimulai
    m_err_latih = np.concatenate([[1],m_err_latih])
    m_err_uji = np.concatenate([[1],m_err_uji])
    m_aku_latih = np.concatenate([[0],m_aku_latih])
    m_aku_uji = np.concatenate([[0],m_aku_uji])

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(x,m_err_latih, marker = 'o', label="data training")
    axs[0].plot(x,m_err_uji, marker = 'x', label="data testing")
    axs[0].set_title('Grafik Error') 
    axs[0].set_ylabel('Error') 
    axs[0].set_xlabel('Epoch ke-')
    axs[0].legend()
    # cetak chart akurasi di subplot 2
    axs[1].plot(x,m_aku_latih, marker = 'o', label="data training")
    axs[1].plot(x,m_aku_uji, marker = 'x', label="data testing")
    axs[1].set_title('Grafik Akurasi') 
    axs[1].set_ylabel('Akurasi (%)') 
    axs[1].set_xlabel('Epoch ke-')
    axs[1].legend()
    plt.show(block=True)
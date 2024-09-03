import numpy as np
import matplotlib.pyplot as plt
from utilitas import get_data, akurasi, cetak_graf
from model import latih, uji
from matplotlib import pyplot as plt 

if __name__=='__main__':
    x_latih,  y_latih = get_data('DataLatih.csv')
    x_uji,  y_uji = get_data('DataUji.csv')
    
    # Melatih model Perceptron, dapat bobot matrix MSE per epoch dan akurasi per epoch
    model, m_err_latih, m_err_uji, m_aku_latih, m_aku_uji = latih(x_latih, y_latih, 0.1, 10)

    # Melakukan prediksi pada menggunakan model terakhir
    y_pred = uji(x_uji, model)
    # Menghitung akurasi model terakhir
    acc = akurasi(y_uji, y_pred)
    print('==============================')
    print(f'Akurasi model: {acc:.2f} %')

    # Menampilkan beberapa prediksi dan label sebenarnya
    print('Bobot model:', model)
    print('Kelas Prediksi:', y_pred)
    print('Kelas Target:', y_uji)
    
    # Memanggil fungsi cetak grafik
    cetak_graf(m_err_latih, m_err_uji, m_aku_uji, m_aku_uji)
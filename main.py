import numpy as np
import matplotlib.pyplot as plt
from utilitas import get_data, akurasi
from model import latih, uji
from matplotlib import pyplot as plt 

if __name__=='__main__':
    x_latih,  y_latih = get_data('DataLatih.csv')
    x_uji,  y_uji = get_data('DataUji.csv')
    
    # Melatih model Perceptron, dapat bobot matrix MSE per epoch dan akurasi per epoch
    model, m_mse, m_acc = latih(x_latih, y_latih, 0.1, 10)

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
    
    # cetak chart error
    x = np.arange(1,11) 
    plt.plot(x,m_mse)
    plt.title('Grafik MSE') 
    plt.ylabel('Error') 
    plt.xlabel('Epoch ke-')
    plt.show()

    # cetak chart akurasi
    x = np.arange(1,11) 
    plt.plot(x,m_acc)
    plt.title('Grafik Akurasi') 
    plt.ylabel('Akurasi (%)') 
    plt.xlabel('Epoch ke-')
    plt.show(block=True)
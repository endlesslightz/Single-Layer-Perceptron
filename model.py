import numpy as np
from utilitas import aktivasi, sigmoid, error, akurasi, loss, get_data

def latih(x, y, learning_rate, epochs):
    # Mencari jml data dan jml fitur berdasarkan shape dari array X
    jml_data, jml_fitur = x.shape
    # +1 untuk bias dan bobot awal semua 0.5
    bobot = np.full(jml_fitur + 1, 0.5)  
    # Menambahkan kolom bias dengan nilai 1 supaya gampang
    X_bias = np.hstack([x, np.ones((jml_data, 1))])  
    # dipakai untuk hitung akurasi dan error per epoch
    x_latih,  y_latih = get_data('DataLatih.csv')  
    x_uji,  y_uji = get_data('DataUji.csv')  
    # persiapan MSE dan akurasi dengan angka 0 sebanyak epoch yang dimasukkan
    err_latih = np.zeros(epochs)
    err_uji = np.zeros(epochs)
    aku_latih = np.zeros(epochs)
    aku_uji = np.zeros(epochs)
    for epoch in range(epochs):
        err=0
        for i in range(jml_data):
            sum = np.dot(X_bias[i], bobot)
            sig = sigmoid(sum)
            err += error(sig,y[i])
            delta = 2*(sig-y[i])*(1-sig)*sig*X_bias[i] # pakai turunan gradient descent
            bobot = bobot - learning_rate * delta
        # pakai bobot dari data terakhir diuji per epoch 
        err_latih[epoch]=loss(y_latih, uji_error(x_latih, bobot))
        err_uji[epoch]=loss(y_uji, uji_error(x_uji, bobot))
        aku_latih[epoch]=akurasi(y_latih, uji(x_latih, bobot))
        aku_uji[epoch]=akurasi(y_uji, uji(x_uji, bobot))
        # print error dan akurasi())
        print(f"-----------------------\nEpoch ke-{epoch}")
        print(f"Error Training: {err_latih[epoch]}")
        print(f"Error Testing: {err_uji[epoch]}")
        print(f"Akurasi Training: {aku_latih[epoch]}")
        print(f"Akurasi Testing: {aku_uji[epoch]}")
    return bobot, err_latih, err_uji, aku_latih, aku_uji

def uji(x, bobot):
    X_bias = np.hstack([x, np.ones((x.shape[0], 1))])  # Menambahkan kolom bias
    linear_output = np.dot(X_bias, bobot)
    return aktivasi(linear_output)


def uji_error(x, bobot):
    X_bias = np.hstack([x, np.ones((x.shape[0], 1))])  # Menambahkan kolom bias
    linear_output = np.dot(X_bias, bobot)
    return linear_output
import numpy as np
from utilitas import aktivasi, sigmoid, error, akurasi, get_data

def latih(x, y, learning_rate, epochs):
    # Mencari jml data dan jml fitur berdasarkan shape dari array X
    jml_data, jml_fitur = x.shape
    # +1 untuk bias dan bobot awal semua 0.5
    bobot = np.full(jml_fitur + 1, 0.5)  
    # Menambahkan kolom bias dengan nilai 1 supaya gampang
    X_bias = np.hstack([x, np.ones((jml_data, 1))])  
    # dipakai untuk hitung akurasi per epoch
    x_valid,  y_valid = get_data('DataUji.csv')  
    # persiapan MSE dan akurasi dengan angka 0 sebanyak epoch yang dimasukkan
    mse = np.zeros(epochs)
    acc = np.zeros(epochs)
    for epoch in range(epochs):
        err=0
        for i in range(jml_data):
            sum = np.dot(X_bias[i], bobot)
            sig = sigmoid(sum)
            # output = aktivasi(sig)
            err += error(sig,y[i])
            delta = 2*(sig-y[i])*(1-sig)*sig*X_bias[i] #pakai gradient descent
            bobot = bobot - learning_rate * delta
        # bobot dari data terakhir diuji 
        y_pred = uji(x_valid, bobot)
        mse[epoch]=err/jml_data
        acc[epoch]=akurasi(y_valid, y_pred)
        # print(acc)
        print(f"Epoch ke-{epoch}")
        print(f"Error: {err/jml_data}")
        print(f"Akurasi: {akurasi(y_valid, y_pred)}")
    return bobot, mse, acc

def uji(x, bobot):
    X_bias = np.hstack([x, np.ones((x.shape[0], 1))])  # Menambahkan kolom bias
    linear_output = np.dot(X_bias, bobot)
    return aktivasi(linear_output)
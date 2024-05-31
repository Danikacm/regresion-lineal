import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def cargar_datos(ruta, columnas):
    datos = pd.read_csv(ruta)
    return datos[columnas]

def regresion_polinomica_grafico(ax, datos, orden_polinomio, columnas, label_error=None):
    X = datos[columnas[0]].values.reshape(-1, 1)
    y = datos[columnas[1]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_features = PolynomialFeatures(degree=orden_polinomio)
    X_train_poly = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)

    if orden_polinomio == usuario_orden or orden_polinomio == mejor_orden:
        ax.scatter(X_test, y_pred, color='black', label=f'Predicciones - Orden {orden_polinomio}')
        ax.scatter(X, y, color='pink', label='Datos reales')
        ax.set_title(f'Regresión Polinómica - Orden {orden_polinomio}')
        ax.set_xlabel('Variable independiente')
        ax.set_ylabel('Variable dependiente')
        ax.legend()

    if label_error is not None:
        label_error.config(text=f"Error cuadrático medio: {mse:.5f}")

    return mse

def limpiar_grafico(ax, label_error=None):
    ax.clear()
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend()
    ax.figure.canvas.draw()

    if label_error is not None:
        label_error.config(text="")

def limpiar(entry_orden, label_error_usuario, label_error_mejor):
    entry_orden.delete(0, tk.END)
    limpiar_grafico(ax_usuario, label_error_usuario)
    limpiar_grafico(ax_mejor, label_error_mejor)

def crear_interfaz_grafica():
    ventana = tk.Tk()
    ventana.title("Regresión Polinómica")

    label_orden = tk.Label(ventana, text="Ingrese el orden del polinomio (1-20):")
    label_orden.grid(row=0, column=0, padx=10, pady=10)

    entry_orden = ttk.Entry(ventana)
    entry_orden.grid(row=0, column=1, padx=10, pady=10)

    btn_mostrar_grafico = ttk.Button(ventana, text="Mostrar gráfico solicitado", command=lambda: mostrar_grafico(int(entry_orden.get())))
    btn_mostrar_grafico.grid(row=1, column=0, columnspan=2, pady=10)

    btn_mostrar_mejor_grafico = ttk.Button(ventana, text="Mostrar gráfico con menor error posible", command=mostrar_mejor_grafico)
    btn_mostrar_mejor_grafico.grid(row=2, column=0, columnspan=2, pady=10)

    btn_limpiar = ttk.Button(ventana, text="Limpiar", command=lambda: limpiar(entry_orden, label_error_usuario, label_error_mejor))
    btn_limpiar.grid(row=3, column=0, columnspan=2, pady=10)

    label_error_usuario = ttk.Label(ventana, text="")
    label_error_usuario.grid(row=4, column=0, pady=10)

    label_error_mejor = ttk.Label(ventana, text="")
    label_error_mejor.grid(row=4, column=1, pady=10)

    return ventana, entry_orden, label_error_usuario, label_error_mejor

def mostrar_grafico(orden):
    global usuario_orden
    if 1 <= orden <= 20:
        limpiar_grafico(ax_usuario, label_error_usuario)
        usuario_orden = orden
        mse_usuario = regresion_polinomica_grafico(ax_usuario, datos, orden, columnas, label_error_usuario)
        canvas_usuario.draw()
        print(f"Error cuadrático medio para polinomio de orden {orden}: {mse_usuario}")
    else:
        print("Error: El orden del polinomio debe estar entre 1 y 20.")

def mostrar_mejor_grafico():
    global mejor_orden
    limpiar_grafico(ax_mejor, label_error_mejor)
    mejor_orden = 0
    menor_error = float('inf')

    for orden in range(1, 21):
        if orden != usuario_orden:
            mse = regresion_polinomica_grafico(ax_mejor, datos, orden, columnas, label_error_mejor)

            if mse is not None and mse < menor_error:
                mejor_orden = orden
                menor_error = mse

    if mejor_orden != 0:
        regresion_polinomica_grafico(ax_mejor, datos, mejor_orden, columnas, label_error_mejor)

    canvas_mejor.draw()
    print(f"El polinomio con orden {mejor_orden} tiene el menor error cuadrático medio: {menor_error}")

if __name__ == "__main__":
    archivo_csv = 'DATASET2AGUA.csv'
    columnas = ['X', 'Y']
    datos = cargar_datos(archivo_csv, columnas)

    ventana, entry_orden, label_error_usuario, label_error_mejor = crear_interfaz_grafica()

    fig_usuario, ax_usuario = plt.subplots()
    canvas_usuario = FigureCanvasTkAgg(fig_usuario, master=ventana)
    canvas_usuario.get_tk_widget().grid(row=5, column=0, pady=10)
    canvas_usuario.draw()

    fig_mejor, ax_mejor = plt.subplots()
    canvas_mejor = FigureCanvasTkAgg(fig_mejor, master=ventana)
    canvas_mejor.get_tk_widget().grid(row=5, column=1, pady=10)
    canvas_mejor.draw()

    ventana.mainloop()

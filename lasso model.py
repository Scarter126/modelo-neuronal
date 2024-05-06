import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Ruta completa donde se encuentran los archivos CSV
directorio = "directory"

# Lista para almacenar las características (X)
caracteristicas = []

# Lista para almacenar el consumo de energía eléctrica (y)
consumo_electrico = []

# Recorrer todos los archivos en el directorio
for archivo in os.listdir(directorio):
    if archivo.endswith(".csv"):
        # Cargar el archivo CSV en un DataFrame de pandas
        datos = pd.read_csv(os.path.join(directorio, archivo), sep=';', header=None, usecols=[0, 1, 2, 3, 4])
        # Filtrar los datos para que solo contengan registros de los años 2020 y 2022
        datos['Fecha'] = pd.to_datetime(datos[1], errors='coerce')
        datos_filtrados = datos[datos[1].str.startswith(('2020', '2022', "2021"))].iloc[:, 2:]

        # Modificar la carga de características y el objetivo
        caracteristicas.extend(datos_filtrados.iloc[:, :-1].values)
        consumo_electrico.extend(datos_filtrados.iloc[:, -1].values)

# Filtrar los datos para que solo contengan registros de los años 2020 y 2022
datos_filtrados = datos[datos[1].str.startswith(('2020', '2022', "2021"))]

# Convertir las fechas a días del año
fechas_dia_del_ano = datos_filtrados['Fecha'].dt.dayofyear.values.reshape(-1, 1)

# Asumiendo que el consumo eléctrico está en la tercera columna (índice 2)
consumo_electrico = datos_filtrados.iloc[:, 2].values

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(fechas_dia_del_ano, consumo_electrico, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construir el modelo neuronal con regularización Lasso (L1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001),
                          input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),
    tf.keras.layers.Dense(1)  # Capa de salida para la regresión
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
mae_test = model.evaluate(X_test_scaled, y_test)[1]
print("MAE en conjunto de prueba:", mae_test)

# Hacer predicciones en el conjunto de prueba
predicciones = model.predict(X_test_scaled)

# Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar predicciones vs valores reales
plt.scatter(y_test, predicciones)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()


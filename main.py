import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from Crypto.Cipher import AES
import os

# Cargar los datos de los activos
data = pd.read_csv('datos_activos.csv')

# Preprocesar los datos
X = data.drop(['riesgo'], axis=1)
y = data['riesgo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de redes neuronales
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Evaluar el modelo en el conjunto de pruebas
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
precision = precision_score(y_test, y_pred_binary)
print('Precisión del modelo:', precision)

# Graficar el valor de los activos en el tiempo
plt.plot(data['tiempo'], data['valor'])
plt.xlabel('Tiempo')
plt.ylabel('Valor de los Activos')
plt.title('Valor de los Activos en el Tiempo')
plt.show()

# Encriptar los datos
key = os.urandom(16)
cipher = AES.new(key, AES.MODE_EAX)
ciphertext, tag = cipher.encrypt_and_digest(data.to_csv().encode())

# Desencriptar los datos
cipher = AES.new(key, AES.MODE_EAX, cipher.nonce)
plaintext = cipher.decrypt_and_verify(ciphertext, tag)

# Convertir los datos a un dataframe de pandas
data_desencriptada = pd.read_csv(io.StringIO(plaintext.decode()))

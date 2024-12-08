import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt

# Función para entrenar el modelo
def train_model():
    # Directorio datos con los que se entrenara y validara la red neuronal
    train_dir = 'C:/Users/Cristian/Desktop/red neuronal/train'
    validation_dir = 'C:/Users/Cristian/Desktop/red neuronal/validation'

    # Creación de un generador de datos con normalización y aumento de datos
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Preparación de los datos de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Preparación de los datos de validación
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Construcción de la CNN con más complejidad
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compilación del modelo
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Callbacks para mejorar el entrenamiento
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Entrenamiento del modelo
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,  # Aumenta el número de épocas
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluación del modelo
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {loss}')
    print(f'Validation Accuracy: {accuracy}')

    # Guardar el modelo entrenado
    model.save('clasificador.h5')
    print("Modelo entrenado y guardado como 'clasificador.h5'")

    # Guardar el historial del entrenamiento y visualizar
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Accuracy de entrenamiento')
    plt.plot(epochs, val_acc, 'b', label='Accuracy de validación')
    plt.title('Accuracy de entrenamiento y validación')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Loss de entrenamiento')
    plt.plot(epochs, val_loss, 'b', label='Loss de validación')
    plt.title('Loss de entrenamiento y validación')
    plt.legend()

    plt.show()

# Función para predecir la imagen
def predict_image(image_path):
    if not os.path.exists('clasificador.h5'):
        print("El modelo no está entrenado. Por favor, entrena el modelo primero.")
        return "Modelo no entrenado"

    model = load_model('clasificador.h5')
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "Perro"
    else:
        return "Gato"

# Función para cargar y predecir imagen desde la interfaz gráfica
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((150, 150))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        prediction = predict_image(file_path)
        result_label.config(text=f"Predicción: {prediction}")

# Función para mostrar la ventana de predicción
def show_prediction_window():
    global panel, result_label

    prediction_window = tk.Toplevel(root)
    prediction_window.title("Clasificador de Gatos y Perros")

    panel = tk.Label(prediction_window)
    panel.pack()

    btn = tk.Button(prediction_window, text="Cargar Imagen", command=upload_and_predict)
    btn.pack()

    result_label = tk.Label(prediction_window, text="Predicción: ", font=("Helvetica", 16))
    result_label.pack()

# Configuración de la interfaz gráfica principal
root = tk.Tk()
root.title("Menu Principal")

train_button = tk.Button(root, text="Entrenar Modelo", command=train_model)
train_button.pack(pady=20)

predict_button = tk.Button(root, text="Hacer Predicción", command=show_prediction_window)
predict_button.pack(pady=20)

root.mainloop()

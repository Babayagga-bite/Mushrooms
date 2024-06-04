# -*- coding: utf-8 -*-
try:
    import RPi.GPIO as GPIO
    on_raspberry_pi = True
except (ImportError, RuntimeError):
    print("RPi.GPIO no está disponible. El script continuará sin la funcionalidad GPIO.")
    on_raspberry_pi = False

from tensorflow.keras.models import load_model  # Usar la versión de TensorFlow de Keras
import numpy as np
import time
import serial
from threading import Timer
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Intentar configurar el puerto serial para la comunicación Bluetooth
try:
    bluetooth = serial.Serial("/dev/rfcomm0", baudrate=9600, timeout=1)  # Agregar un timeout para evitar bloqueos
    bluetooth_available = True
except serial.SerialException as e:
    print(f"No se pudo abrir el puerto serial: {e}")
    bluetooth_available = False

if on_raspberry_pi:
    # Configuración de los pines GPIO
    GPIO.setmode(GPIO.BOARD)
    pin_to_circuit = 7

# Verificar la existencia del archivo del modelo
model_path = 'pegasus_detection_model.h5'
if os.path.exists(model_path):
    try:
        model = load_model(model_path, compile=False)  # Usar compile=False para evitar problemas de compatibilidad
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None
else:
    print(f"El archivo del modelo no se encuentra en la ruta: {model_path}")
    model = None

def rc_time(pin_to_circuit):
    count = 0
    if on_raspberry_pi:
        # Salida de pin a bajo
        GPIO.setup(pin_to_circuit, GPIO.OUT)
        GPIO.output(pin_to_circuit, GPIO.LOW)
        time.sleep(0.1)

        # Cambiar el pin de entrada y contar hasta que vaya a HIGH
        GPIO.setup(pin_to_circuit, GPIO.IN)
        # Mientras el pin de entrada esté bajo
        while GPIO.input(pin_to_circuit) == GPIO.LOW:
            count += 1

    return count

def stop_bluetooth_communication():
    # Enviar una señal para detener la comunicación Bluetooth
    if bluetooth_available:
        bluetooth.write(b"0")

# Funciones adicionales del segundo script
class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0

def get_mushroom_name():
    return "Sample Mushroom"

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = self._build_model()
    
    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.input_size, activation='relu'))
        model.add(Dense(self.output_size, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=10):
        self.model.fit(X, y, epochs=epochs)

def detect_pegasus(logs, model_path):
    try:
        model = load_model(model_path, compile=False)  # Usar compile=False para evitar problemas de compatibilidad
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    # Placeholder logic for detection
    for log in logs:
        if "Pegasus" in log:
            return True
    return False

def run_gui(mushroom):
    window = tk.Tk()
    window.title("Mushroom Monitoring System")
    
    def start_bluetooth_scan():
        log_text.insert(tk.END, "Starting Bluetooth scan...\n")
        # Placeholder for starting Bluetooth scan
        mushroom.total_devices += 1
        mushroom.fungi_type_a += 1
        update_log()

    def stop_bluetooth_scan():
        log_text.insert(tk.END, "Stopping Bluetooth scan...\n")
        # Placeholder for stopping Bluetooth scan

    def start_wifi_scan():
        log_text.insert(tk.END, "Starting WiFi scan...\n")
        # Placeholder for starting WiFi scan
        mushroom.fungi_type_b += 1
        update_log()

    def update_log():
        log_text.insert(tk.END, f"Total Devices: {mushroom.total_devices}\n")
        log_text.insert(tk.END, f"Fungi Type A: {mushroom.fungi_type_a}\n")
        log_text.insert(tk.END, f"Fungi Type B: {mushroom.fungi_type_b}\n")

    def detect_pegasus_logs():
        logs = log_text.get(1.0, tk.END).splitlines()
        model_path = "pegasus_detection_model.h5"
        pegasus_detected = detect_pegasus(logs, model_path)
        if pegasus_detected:
            messagebox.showinfo("Pegasus Detection", "Pegasus detected in the logs!")
        else:
            messagebox.showinfo("Pegasus Detection", "No Pegasus detected in the logs.")

    scan_bluetooth_button = tk.Button(window, text="Start Bluetooth Scan", command=start_bluetooth_scan)
    scan_bluetooth_button.pack()

    stop_bluetooth_button = tk.Button(window, text="Stop Bluetooth Scan", command=stop_bluetooth_scan)
    stop_bluetooth_button.pack()

    scan_wifi_button = tk.Button(window, text="Start WiFi Scan", command=start_wifi_scan)
    scan_wifi_button.pack()

    detect_pegasus_button = tk.Button(window, text="Detect Pegasus", command=detect_pegasus_logs)
    detect_pegasus_button.pack()

    log_text = scrolledtext.ScrolledText(window, width=50, height=10)
    log_text.pack()

    window.mainloop()

if __name__ == "__main__":
    mushroom_name = get_mushroom_name()
    mushroom = Mushroom(mushroom_name)
    print(f"Nombre del hongo: {mushroom.name}")

    input_size = 1
    hidden_size = 5
    output_size = 1
    mushroom.nn = MiniNeuralNetwork(input_size, hidden_size, output_size)

    # Entrenar la red neuronal con datos de ejemplo (X, y)
    X = np.random.rand(1000, 1)
    y = np.random.randint(0, 2, size=(1000, 1))
    mushroom.nn.train(X, y, epochs=10)

    logs = ["Ejemplo de log de Pegasus..."]
    model_path = "pegasus_detection_model.h5"
    pegasus_detected = detect_pegasus(logs, model_path)
    print(f"Pegasus detectado: {pegasus_detected}")

    # Ejecutar GUI
    run_gui(mushroom)


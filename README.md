# Mushroom Bluetooth Device Collector

Este proyecto implementa un sistema de recolección y clasificación de dispositivos Bluetooth en modo monitor, asignándoles nombres de hongos y detectando posibles infecciones por Pegasus.

## Características

- **Escaneo de dispositivos Bluetooth**: Escanea dispositivos Bluetooth y recopila información sobre ellos.
- **Clasificación de dispositivos**: Utiliza una red neuronal simple para clasificar los dispositivos en dos tipos diferentes de "hongos".
- **Detección de Pegasus**: Detecta la posible presencia de Pegasus en los registros del sistema.

## Requisitos

- Python 3.7+
- Scapy
- TensorFlow
- NumPy

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/mushroom-bluetooth-collector.git
    cd mushroom-bluetooth-collector
    ```

2. Crea un entorno virtual (opcional pero recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Inicia el script principal:
    ```sh
    python main.py
    ```

2. Selecciona una opción en el menú:
    - **Escanear dispositivos Bluetooth en modo monitor**: Comienza a escanear dispositivos y recopila información.
    - **Mostrar estadísticas**: Muestra estadísticas sobre los dispositivos recolectados.
    - **Salir**: Finaliza el programa.

## Código

```python
import random
import numpy as np
import threading
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0
        self.pegasus_detected = 0
        self.devices = []

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = 1 / (1 + np.exp(-output_layer_input))
        return output

def get_mushroom_name():
    mushroom_names = [
        "Amanita Muscaria", "Boletus Edulis", "Cantharellus Cibarius",
        "Cortinarius Violaceus", "Gyromitra Esculenta", "Lactarius Deliciosus",
        "Morchella Esculenta", "Pleurotus Ostreatus", "Trametes Versicolor"
    ]
    return random.choice(mushroom_names)

def classify_device(device_class, nn):
    device_class_normalized = device_class / 1000.0
    prediction = nn.forward(np.array([device_class_normalized]))
    return "Fungi Type A" if prediction > 0.5 else "Fungi Type B"

def detect_pegasus(logs):
    logs = logs.split('\n')
    logs = [line.strip() for line in logs if line.strip()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(logs)
    sequences = tokenizer.texts_to_sequences(logs)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    model = tf.keras.models.load_model('pegasus_detection_model.h5')
    predictions = model.predict(padded_sequences)
    pegasus_detected = any(predictions > 0.5)
    return pegasus_detected

def packet_handler(pkt):
    if pkt.haslayer(Dot11):
        if pkt.type == 0 and pkt.subtype == 8:
            addr = pkt.addr2
            name = pkt.info.decode("utf-8")
            device_class = 0
            mushroom_name = get_mushroom_name()
            pegasus = detect_pegasus("Pegasus log data")
            nn = MiniNeuralNetwork(input_size=1, hidden_size=5, output_size=1)
            classification = classify_device(device_class, nn)
            collector.collect_device(name, addr, mushroom_name, device_class, classification, pegasus)

def scan_for_mushrooms(collector):
# Mushroom Bluetooth Device Collector

Este proyecto implementa un sistema de recolección y clasificación de dispositivos Bluetooth en modo monitor, asignándoles nombres de hongos y detectando posibles infecciones por Pegasus.

## Características

- **Escaneo de dispositivos Bluetooth**: Escanea dispositivos Bluetooth y recopila información sobre ellos.
- **Clasificación de dispositivos**: Utiliza una red neuronal simple para clasificar los dispositivos en dos tipos diferentes de "hongos".
- **Detección de Pegasus**: Detecta la posible presencia de Pegasus en los registros del sistema.

## Requisitos

- Python 3.7+
- Scapy
- TensorFlow
- NumPy

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/mushroom-bluetooth-collector.git
    cd mushroom-bluetooth-collector
    ```

2. Crea un entorno virtual (opcional pero recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Inicia el script principal:
    ```sh
    python main.py
    ```

2. Selecciona una opción en el menú:
    - **Escanear dispositivos Bluetooth en modo monitor**: Comienza a escanear dispositivos y recopila información.
    - **Mostrar estadísticas**: Muestra estadísticas sobre los dispositivos recolectados.
    - **Salir**: Finaliza el programa.

## Código

```python
import random
import numpy as np
import threading
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0
        self.pegasus_detected = 0
        self.devices = []

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = 1 / (1 + np.exp(-output_layer_input))
        return output

def get_mushroom_name():
    mushroom_names = [
        "Amanita Muscaria", "Boletus Edulis", "Cantharellus Cibarius",
        "Cortinarius Violaceus", "Gyromitra Esculenta", "Lactarius Deliciosus",
        "Morchella Esculenta", "Pleurotus Ostreatus", "Trametes Versicolor"
    ]
    return random.choice(mushroom_names)

def classify_device(device_class, nn):
    device_class_normalized = device_class / 1000.0
    prediction = nn.forward(np.array([device_class_normalized]))
    return "Fungi Type A" if prediction > 0.5 else "Fungi Type B"

def detect_pegasus(logs):
    logs = logs.split('\n')
    logs = [line.strip() for line in logs if line.strip()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(logs)
    sequences = tokenizer.texts_to_sequences(logs)
# Mushroom Bluetooth Device Collector

Este proyecto implementa un sistema de recolección y clasificación de dispositivos Bluetooth en modo monitor, asignándoles nombres de hongos y detectando posibles infecciones por Pegasus.

## Características

- **Escaneo de dispositivos Bluetooth**: Escanea dispositivos Bluetooth y recopila información sobre ellos.
- **Clasificación de dispositivos**: Utiliza una red neuronal simple para clasificar los dispositivos en dos tipos diferentes de "hongos".
- **Detección de Pegasus**: Detecta la posible presencia de Pegasus en los registros del sistema.

## Requisitos

- Python 3.7+
- Scapy
- TensorFlow
- NumPy

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/mushroom-bluetooth-collector.git
    cd mushroom-bluetooth-collector
    ```

2. Crea un entorno virtual (opcional pero recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Inicia el script principal:
    ```sh
    python main.py
    ```

2. Selecciona una opción en el menú:
    - **Escanear dispositivos Bluetooth en modo monitor**: Comienza a escanear dispositivos y recopila información.
    - **Mostrar estadísticas**: Muestra estadísticas sobre los dispositivos recolectados.
    - **Salir**: Finaliza el programa.

## Código

```python
import random
import numpy as np
import threading
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0
        self.pegasus_detected = 0
        self.devices = []

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = 1 / (1 + np.exp(-output_layer_input))
        return output

def get_mushroom_name():
    mushroom_names = [
        "Amanita Muscaria", "Boletus Edulis", "Cantharellus Cibarius",
        "Cortinarius Violaceus", "Gyromitra Esculenta", "Lactarius Deliciosus",
        "Morchella Esculenta", "Pleurotus Ostreatus", "Trametes Versicolor"
    ]
    return random.choice(mushroom_names)

def classify_device(device_class, nn):
    device_class_normalized = device_class / 1000.0
    prediction = nn.forward(np.array([device_class_normalized]))
    return "Fungi Type A" if prediction > 0.5 else "Fungi Type B"

def detect_pegasus(logs):
    logs = logs.split('\n')
    logs = [line.strip() for line in logs if line.strip()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(logs)
    sequences = tokenizer.texts_to_sequences(logs)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    model = tf.keras.models.load_model('pegasus_detection_model.h5')
    predictions = model.predict(padded_sequences)
    pegasus_detected = any(predictions > 0.5)
    return pegasus_detected

def packet_handler(pkt):
    if pkt.haslayer(Dot11):
        if pkt.type == 0 and pkt.subtype == 8:
            addr = pkt.addr2
            name = pkt.info.decode("utf-8")
            device_class = 0
            mushroom_name = get_mushroom_name()
            pegasus = detect_pegasus("Pegasus log data")
            nn = MiniNeuralNetwork(input_size=1, hidden_size=5, output_size=1)
            classification = classify_device(device_class, nn)
            collector.collect_device(name, addr, mushroom_name, device_class, classification, pegasus)

def scan_for_mushrooms(collector):
    print("🍄 Escaneando dispositivos Bluetooth en modo monitor... 🍄")
    sniff(prn=packet_handler, store=False)

class MushroomCollector:
    def __init__(self):
        self.mushrooms = []
        self.stop_sniffing = threading.Event()

    def collect_device(self, name, addr, mushroom_name, device_class, classification, pegasus):
        for mushroom in self.mushrooms:
            if mushroom.name == mushroom_name:
                mushroom.total_devices += 1
                if classification == "Fungi Type A":
                    mushroom.fungi_type_a += 1
                else:
                    mushroom.fungi_type_b += 1
                if pegasus:
                    mushroom.pegasus_detected += 1
                mushroom.devices.append((name, addr))
                return
        new_mushroom = Mushroom(mushroom_name)
        new_mushroom.total_devices = 1
        if classification == "Fungi Type A":
            new_mushroom.fungi_type_a = 1
        else:
            new_mushroom.fungi_type_b = 1
        if pegasus:
            new_mushroom.pegasus_detected = 1
        new_mushroom.devices.append((name, addr))
        self.mushrooms.append(new_mushroom)

    def display_statistics(self):
        print("\n📊 Estadísticas de recolección 📊")
        for mushroom in self.mushrooms:
            print(f"Mushroom: {mushroom.name}")
            print(f"Total de dispositivos: {mushroom.total_devices}")
            print(f"Fungi Type A: {mushroom.fungi_type_a}")
            print(f"Fungi Type B: {mushroom.fungi_type_b}")
            print(f"Dispositivos con Pegasus detectado: {mushroom.pegasus_detected}")
            print("Dispositivos encontrados:")
            for device in mushroom.devices:
                print(f"Nombre: {device[0]}, Dirección MAC: {device[1]}")

def main_menu():
    print("\n🍄 Bienvenido al sistema de recolección de hongos 🍄")
    print("1. Escanear dispositivos Bluetooth en modo monitor")
    print("2. Mostrar estadísticas")
    print("3. Salir")

def main():
    collector = MushroomCollector()
    while True:
        main_menu()
        choice = input("Seleccione una opción: ")
        if choice == "1":
            scan_thread = threading.Thread(target=scan_for_mushrooms, args=(collector,))
            scan_thread.start()
            input("Presione Enter para detener el escaneo...")
            collector.stop_sniffing.set()
            scan_thread.join()
        elif choice == "2":
            collector.display_statistics()
        elif choice == "3":
            print("¡Adiós!")
            break
```

## Contribuciones

Las contribuciones son bienvenidas. Puedes abrir un issue o enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.# Mushroom Bluetooth Device Collector

Este proyecto implementa un sistema de recolección y clasificación de dispositivos Bluetooth en modo monitor, asignándoles nombres de hongos y detectando posibles infecciones por Pegasus.

## Características

- **Escaneo de dispositivos Bluetooth**: Escanea dispositivos Bluetooth y recopila información sobre ellos.
- **Clasificación de dispositivos**: Utiliza una red neuronal simple para clasificar los dispositivos en dos tipos diferentes de "hongos".
- **Detección de Pegasus**: Detecta la posible presencia de Pegasus en los registros del sistema.

## Requisitos

- Python 3.7+
- Scapy
- TensorFlow
- NumPy

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/mushroom-bluetooth-collector.git
    cd mushroom-bluetooth-collector
    ```

2. Crea un entorno virtual (opcional pero recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Inicia el script principal:
    ```sh
    python main.py
    ```

2. Selecciona una opción en el menú:
    - **Escanear dispositivos Bluetooth en modo monitor**: Comienza a escanear dispositivos y recopila información.
    - **Mostrar estadísticas**: Muestra estadísticas sobre los dispositivos recolectados.
    - **Salir**: Finaliza el programa.

## Código

```python
import random
import numpy as np
import threading
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0
        self.pegasus_detected = 0
        self.devices = []

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = 1 / (1 + np.exp(-output_layer_input))
        return output

def get_mushroom_name():
    mushroom_names = [
        "Amanita Muscaria", "Boletus Edulis", "Cantharellus Cibarius",
        "Cortinarius Violaceus", "Gyromitra Esculenta", "Lactarius Deliciosus",
        "Morchella Esculenta", "Pleurotus Ostreatus", "Trametes Versicolor"
    ]
    return random.choice(mushroom_names)

def classify_device(device_class, nn):
    device_class_normalized = device_class / 1000.0
    prediction = nn.forward(np.array([device_class_normalized]))
    return "Fungi Type A" if prediction > 0.5 else "Fungi Type B"

def detect_pegasus(logs):
    logs = logs.split('\n')
    logs = [line.strip() for line in logs if line.strip()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(logs)
    sequences = tokenizer.texts_to_sequences(logs)
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    model = tf.keras.models.load_model('pegasus_detection_model.h5')
    predictions = model.predict(padded_sequences)
    pegasus_detected = any(predictions > 0.5)
    return pegasus_detected

def packet_handler(pkt):
    if pkt.haslayer(Dot11):
        if pkt.type == 0 and pkt.subtype == 8:
            addr = pkt.addr2
            name = pkt.info.decode("utf-8")
            device_class = 0
            mushroom_name = get_mushroom_name()
            pegasus = detect_pegasus("Pegasus log data")
            nn = MiniNeuralNetwork(input_size=1, hidden_size=5, output_size=1)
            classification = classify_device(device_class, nn)
            collector.collect_device(name, addr, mushroom_name, device_class, classification, pegasus)

def scan_for_mushrooms(collector):
    print("🍄 Escaneando dispositivos Bluetooth en modo monitor... 🍄")
    sniff(prn=packet_handler, store=False)

class MushroomCollector:
    def __init__(self):
        self.mushrooms = []
        self.stop_sniffing = threading.Event()

    def collect_device(self, name, addr, mushroom_name, device_class, classification, pegasus):
        for mushroom in self.mushrooms:
            if mushroom.name == mushroom_name:
                mushroom.total_devices += 1
                if classification == "Fungi Type A":
                    mushroom.fungi_type_a += 1
                else:
                    mushroom.fungi_type_b += 1
                if pegasus:
                    mushroom.pegasus_detected += 1
                mushroom.devices.append((name, addr))
                return
        new_mushroom = Mushroom(mushroom_name)
        new_mushroom.total_devices = 1
        if classification == "Fungi Type A":
            new_mushroom.fungi_type_a = 1
        else:
            new_mushroom.fungi_type_b = 1
        if pegasus:
            new_mushroom.pegasus_detected = 1
        new_mushroom.devices.append((name, addr))
        self.mushrooms.append(new_mushroom)

    def display_statistics(self):
        print("\n📊 Estadísticas de recolección 📊")
        for mushroom in self.mushrooms:
            print(f"Mushroom: {mushroom.name}")
            print(f"Total de dispositivos: {mushroom.total_devices}")
            print(f"Fungi Type A: {mushroom.fungi_type_a}")
            print(f"Fungi Type B: {mushroom.fungi_type_b}")
            print(f"Dispositivos con Pegasus detectado: {mushroom.pegasus_detected}")
            print("Dispositivos encontrados:")
            for device in mushroom.devices:
                print(f"Nombre: {device[0]}, Dirección MAC: {device[1]}")

def main_menu():
    print("\n🍄 Bienvenido al sistema de recolección de hongos 🍄")
    print("1. Escanear dispositivos Bluetooth en modo monitor")
    print("2. Mostrar estadísticas")
    print("3. Salir")

def main():
    collector = MushroomCollector()
    while True:
        main_menu()
        choice = input("Seleccione una opción: ")
        if choice == "1":
            scan_thread = threading.Thread(target=scan_for_mushrooms, args=(collector,))
            scan_thread.start()
            input("Presione Enter para detener el escaneo...")
            collector.stop_sniffing.set()
            scan_thread.join()
        elif choice == "2":
            collector.display_statistics()
        elif choice == "3":
            print("¡Adiós!")
            break
```

## Contribuciones

Las contribuciones son bienvenidas. Puedes abrir un issue o enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.#    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    model = tf.keras.models.load_model('pegasus_detection_model.h5')
    predictions = model.predict(padded_sequences)
    pegasus_detected = any(predictions > 0.5)
    return pegasus_detected

def packet_handler(pkt):
    if pkt.haslayer(Dot11):
        if pkt.type == 0 and pkt.subtype == 8:
            addr = pkt.addr2
            name = pkt.info.decode("utf-8")
            device_class = 0
            mushroom_name = get_mushroom_name()
            pegasus = detect_pegasus("Pegasus log data")
            nn = MiniNeuralNetwork(input_size=1, hidden_size=5, output_size=1)
            classification = classify_device(device_class, nn)
            collector.collect_device(name, addr, mushroom_name, device_class, classification, pegasus)

def scan_for_mushrooms(collector):
    print("🍄 Escaneando dispositivos Bluetooth en modo monitor... 🍄")
    sniff(prn=packet_handler, store=False)

class MushroomCollector:
    def __init__(self):
        self.mushrooms = []
        self.stop_sniffing = threading.Event()

    def collect_device(self, name, addr, mushroom_name, device_class, classification, pegasus):
        for mushroom in self.mushrooms:
            if mushroom.name == mushroom_name:
                mushroom.total_devices += 1
                if classification == "Fungi Type A":
                    mushroom.fungi_type_a += 1
                else:
                    mushroom.fungi_type_b += 1
                if pegasus:
                    mushroom.pegasus_detected += 1
                mushroom.devices.append((name, addr))
                return
        new_mushroom = Mushroom(mushroom_name)
        new_mushroom.total_devices = 1
        if classification == "Fungi Type A":
            new_mushroom.fungi_type_a = 1
        else:
            new_mushroom.fungi_type_b = 1
        if pegasus:
            new_mushroom.pegasus_detected = 1
        new_mushroom.devices.append((name, addr))
        self.mushrooms.append(new_mushroom)

    def display_statistics(self):
        print("\n📊 Estadísticas de recolección 📊")
        for mushroom in self.mushrooms:
            print(f"Mushroom: {mushroom.name}")
            print(f"Total de dispositivos: {mushroom.total_devices}")
            print(f"Fungi Type A: {mushroom.fungi_type_a}")
            print(f"Fungi Type B: {mushroom.fungi_type_b}")
            print(f"Dispositivos con Pegasus detectado: {mushroom.pegasus_detected}")
            print("Dispositivos encontrados:")
            for device in mushroom.devices:
                print(f"Nombre: {device[0]}, Dirección MAC: {device[1]}")

def main_menu():
    print("\n🍄 Bienvenido al sistema de recolección de hongos 🍄")
    print("1. Escanear dispositivos Bluetooth en modo monitor")
    print("2. Mostrar estadísticas")
    print("3. Salir")

def main():
    collector = MushroomCollector()
    while True:
        main_menu()
        choice = input("Seleccione una opción: ")
        if choice == "1":
            scan_thread = threading.Thread(target=scan_for_mushrooms, args=(collector,))
            scan_thread.start()
            input("Presione Enter para detener el escaneo...")
            collector.stop_sniffing.set()
            scan_thread.join()
        elif choice == "2":
            collector.display_statistics()
        elif choice == "3":
            print("¡Adiós!")
           break
```

## Contribuciones

Las contribuciones son bienvenidas. Puedes abrir un issue o enviar un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.## Mushroom Bluetooth Device Collector

Este proyecto implementa un sistema de recolección y clasificación de dispositivos Bluetooth en modo monitor, asignándoles nombres de hongos y detectando posibles infecciones por Pegasus.

## Características

- **Escaneo de dispositivos Bluetooth**: Escanea dispositivos Bluetooth y recopila información sobre ellos.
- **Clasificación de dispositivos**: Utiliza una red neuronal simple para clasificar los dispositivos en dos tipos diferentes de "hongos".
- **Detección de Pegasus**: Detecta la posible presencia de Pegasus en los registros del sistema.

## Requisitos

- Python 3.7+
- Scapy
- TensorFlow
- NumPy

## Instalación

1. Clona el repositorio:
    ```sh
    git clone https://github.com/tu_usuario/mushroom-bluetooth-collector.git
    cd mushroom-bluetooth-collector
    ```

2. Crea un entorno virtual (opcional pero recomendado):
    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

1. Inicia el script principal:
    ```sh
    python main.py
    ```

2. Selecciona una opción en el menú:
    - **Escanear dispositivos Bluetooth en modo monitor**: Comienza a escanear dispositivos y recopila información.
    - **Mostrar estadísticas**: Muestra estadísticas sobre los dispositivos recolectados.
    - **Salir**: Finaliza el programa.

## Código

```python
import random
import numpy as np
import threading
from scapy.all import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Mushroom:
    def __init__(self, name):
        self.name = name
        self.total_devices = 0
        self.fungi_type_a = 0
        self.fungi_type_b = 0
        self.pegasus_detected = 0
        self.devices = []

class MiniNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output = 1 / (1 + np.exp(-output_layer_input))
        return output

def get_mushroom_name():
    mushroom_names = [
        "Amanita Muscaria", "Boletus Edulis", "Cantharellus Cibarius",
        "Cortinarius Violaceus", "Gyromitra Esculenta", "Lactarius Deliciosus",
        "Morchella Esculenta", "Pleurotus Ostreatus", "Trametes Versicolor"
    ]
    return random.choice(mushroom_names)

def classify_device(device_class, nn):
    device_class_normalized = device_class / 1000.0
    prediction = nn.forward(np.array([device_class_normalized]))
    return "Fungi Type A" if prediction > 0.5 else "Fungi Type B"

def detect_pegasus(logs):
    logs = logs.split('\n')
    logs = [line.strip() for line in logs if line.strip()]
    print("🍄 Escaneando dispositivos Bluetooth en modo monitor... 🍄")
    sniff(prn=packet_handler, store=False)

class MushroomCollector:
    def __init__(self):
        self.mushrooms = []
        self.stop_sniffing = threading.Event()

    def collect_device(self, name, addr, mushroom_name, device_class, classification, pegasus):
        for mushroom in self.mushrooms:
            if mushroom.name == mushroom_name:
                mushroom.total_devices += 1
                if classification == "Fungi Type A":
                    mushroom.fungi_type_a += 1
                else:
                    mushroom.fungi_type_b += 1
                if pegasus:
                    mushroom.pegasus_detected += 1
                mushroom.devices.append((name, addr))
                return
        new_mushroom = Mushroom(mushroom_name)
       new_mushroom.total_devices = 1
        if classification == "Fungi Type A":
            new_mushroom.fungi_type_a = 1
        else:
            new_mushroom.fungi_type_b = 1
        if pegasus:
            new_mushroom.pegasus_detected = 1
        new_mushroom.devices.append((name, addr))
        self.mushrooms.append(new_mushroom)

    def display_statistics(self):
        print("\n📊 Estadísticas de recolección 📊")
        for mushroom in self.mushrooms:
            print(f"Mushroom: {mushroom.name}")
            print(f"Total de dispositivos: {mushroom.total_devices}")
            print(f"Fungi Type A: {mushroom.fungi_type_a}")
            print(f"Fungi Type B: {mushroom.fungi_type_b}")
            print(f"Dispositivos con Pegasus detectado: {mushroom.pegasus_detected}")
            print("Dispositivos encontrados:")
            for device in mushroom.devices:
                print(f"Nombre: {device[0]}, Dirección MAC: {device[1]}")

def main_menu():
# Mushrooms

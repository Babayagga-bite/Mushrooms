#!/bin/bash

# Nombre del entorno virtual
VENV_DIR="myenv"

# Comprobar si python3-venv está instalado
if ! dpkg -s python3-venv &> /dev/null; then
    echo "python3-venv no está instalado. Instalándolo ahora..."
    sudo apt update
    sudo apt install -y python3-venv
fi

# Comprobar si las herramientas necesarias para Bluetooth están instaladas
if ! dpkg -s libbluetooth-dev &> /dev/null; then
    echo "libbluetooth-dev no está instalado. Instalándolo ahora..."
    sudo apt install -y libbluetooth-dev
fi

# Crear el entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv $VENV_DIR
fi

# Activar el entorno virtual
echo "Activando entorno virtual..."
source $VENV_DIR/bin/activate

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# Instalar paquetes necesarios
echo "Instalando paquetes necesarios..."
pip install --upgrade pyOpenSSL tensorflow scapy tk bleak RPi.GPIO opencv-python pyserial

# Verificar si la instalación de paquetes fue exitosa
if [ $? -ne 0 ]; then
    echo "Hubo un error al instalar los paquetes. Verifica tu conexión a internet y los permisos."
    deactivate
    exit 1
fi

# Ejecutar el script de Python
echo "Ejecutando script de Python..."
python mushrooms1.3.py

# Verificar si el script de Python se ejecutó correctamente
if [ $? -ne 0 ]; then
    echo "Hubo un error al ejecutar el script de Python."
    deactivate
    exit 1
fi

# Desactivar el entorno virtual
echo "Desactivando entorno virtual..."
deactivate

echo "Proceso completado exitosamente."


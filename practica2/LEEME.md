# Computacion Paralela y Distribuida

## Práctica 1 - Pthread

Hay dos formas de ejecutar el programa para aplicar el filtro **(Sobel)** sobre imagenes.

### Ejecución automatizada para pruebas de rendimiento
```
 ./script.sh
```
### Ejecución manual del programa

#### Compilación

```
sudo gcc image-effect.c -o my-effect -lm -pthread
```

#### Ejecución
```
./my-effect input_image output_image kernel_parameter nThreads
```
Donde:
**input_image:**  Es el nombre o ubicación de la imagen a procesar  
**output_image:**  Es el nombre o ubicación de la imagen resultante al aplicar el filtro  
**kernel_parameter:**  Parámetro que varía entre 0 y 5 e indica el kernel de convolución a utilizar  
**nThreads:**  Parámetro que indica el número de hilos de ejecución  

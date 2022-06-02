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

* -1, 0, 1, -2, 0, 2, -1, 0, 1     // Border detection (Sobel) - 0
*  1, -2, 1, -2, 5, -2, 1, -2, 1  // Sharpen - 1
*  1, 1, 1, 1, -2, 1, -1, -1, -1   // Norte - 2
* -1, 1, 1, -1, -2, 1, -1, 1, 1  // Este - 3
* -1, -1, 0, -1, 0, 1, 0, 1, 1    // Estampado en relieve  - 4 
* -1, -1, -1, -1, 8, -1, -1, -1, -1 // Border detection (Sobel2) - 5

**nThreads:**  Parámetro que indica el número de hilos de ejecución  

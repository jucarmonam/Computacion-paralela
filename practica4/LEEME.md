# Computacion Paralela y Distribuida

## Práctica 3 - CUDA

Hay dos formas de ejecutar el programa para aplicar el filtro **(Sobel)** sobre imagenes.

### Ejecución automatizada para pruebas de rendimiento

Al ejecutar de forma automatizada se realiza un análisis de las características del sistema y se realizan pruebas dependiendo del número de multiprocesadores y núcleos de la GPU.

```
 ./script.sh
```
### Ejecución manual del programa

#### Compilación

```
sudo nvcc image-effect.cu -o my-effect -w
```

#### Ejecución
```
./my-effect input_image output_image kernel_parameter nBlocks nThreads

```

Donde:
**input_image:**  Es el nombre o ubicación de la imagen a procesar  
**output_image:**  Es el nombre o ubicación de la imagen resultante al aplicar el filtro  
**kernel_parameter:**  Parámetro que varía entre 0 y 5 e indica el kernel de convolución a utilizar  
**nBlocks:**  Parámetro que indica el número de bloques de ejecución  
**nThreads:**  Parámetro que indica el número de hilos de ejecución  

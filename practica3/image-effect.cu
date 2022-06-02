/**
 * @file main.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-02-06
 * @copyright Copyright (c) 2022
 */

/*Para la lectura y escritura de imagenes se usan las librerias stb_image*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define R_ARGS 5
#define EXPORT_QUALITY 100

void initializeMatrix(int *matR, int *matG, int *matB, int width, int height, int channels, unsigned char *img)
{
    int i = 0, j = 0;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            *(matR + (i * width + j)) = *(img + (channels * (i * width + j)));
            *(matG + ((i * width) + j)) = *(img + (channels * (i * width + j) + 1));
            *(matB + ((i * width) + j)) = *(img + (channels * (i * width + j) + 2));
        }
    }
}

void joinMatrix(int *matR, int *matG, int *matB, int width, int height, int channels, unsigned char *resImg, unsigned char *img)
{
    int i = 0, j = 0;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            *(resImg + (channels * (i * width + j))) = *(matR + (i * width + j));
            *(resImg + (channels * (i * width + j) + 1)) = *(matG + (i * width + j));
            *(resImg + (channels * (i * width + j) + 2)) = *(matB + (i * width + j));
            if (channels == 4)
                *(resImg + (channels * (i * width + j) + 3)) = *(img + (channels * (i * width + j) + 3));
        }
    }
}

__global__ void applyFilter(int *matR, int *matG, int *matB, int *rMatR, int *rMatG, int *rMatB, int width, int height, int nThreads, int *ker)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    /*Variables necesarias para la convolución*/
    int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
    int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
    int conv = 0;

    /*Calcular la posición inicial en términos de i y j*/
    int i = (startPos / width), j = (startPos % width);

    /*Realizar la convolucion*/
    for (startPos; startPos <= endPos; startPos++)
    {
        /*Ignorar la convolucion en los bordes*/
        if (i > 0 && i < height - 1 && j > 0 && j < width - 1)
        {
            conv = (*(ker) * *(matR + ((i - 1) * width + j - 1)) + *(ker + 1) * *(matR + ((i - 1) * width + j)) + *(ker + 2) * *(matR + ((i - 1) * width + j + 1)) + *(ker + 3) * *(matR + (i * width + j - 1)) + *(ker + 4) * *(matR + (i * width + j)) + *(ker + 5) * *(matR + (i * width + j + 1)) + *(ker + 6) * *(matR + ((i + 1) * width + j - 1)) + *(ker + 7) * *(matR + ((i + 1) * width + j)) + *(ker + 8) * *(matR + ((i + 1) * width + j + 1))) % 255;
            *(rMatR + (i * width + j)) = conv < 0 ? 0 : conv;

            /*Convolucion para el canal G*/
            conv = (*(ker) * *(matG + ((i - 1) * width + j - 1)) + *(ker + 1) * *(matG + ((i - 1) * width + j)) + *(ker + 2) * *(matG + ((i - 1) * width + j + 1)) + *(ker + 3) * *(matG + (i * width + j - 1)) + *(ker + 4) * *(matG + (i * width + j)) + *(ker + 5) * *(matG + (i * width + j + 1)) + *(ker + 6) * *(matG + ((i + 1) * width + j - 1)) + *(ker + 7) * *(matG + ((i + 1) * width + j)) + *(ker + 8) * *(matG + ((i + 1) * width + j + 1))) % 255;
            *(rMatG + (i * width + j)) = conv < 0 ? 0 : conv;

            /*Convolucion para el canal B*/
            conv = (*(ker) * *(matB + ((i - 1) * width + j - 1)) + *(ker + 1) * *(matB + ((i - 1) * width + j)) + *(ker + 2) * *(matB + ((i - 1) * width + j + 1)) + *(ker + 3) * *(matB + (i * width + j - 1)) + *(ker + 4) * *(matB + (i * width + j)) + *(ker + 5) * *(matB + (i * width + j + 1)) + *(ker + 6) * *(matB + ((i + 1) * width + j - 1)) + *(ker + 7) * *(matB + ((i + 1) * width + j)) + *(ker + 8) * *(matB + ((i + 1) * width + j + 1))) % 255;
            *(rMatB + (i * width + j)) = conv < 0 ? 0 : conv;
        }
        j += 1;
        if (j == width)
        {
            i += 1;
            j = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    /*Declarar los string de lectura y escritura*/
    char *loadPath, *savePath;
    /*Variable para escoger el kernel*/
    int argKer = 0;
    /*Declaración de variable para la escritura del archivo*/
    FILE *fp;
    /*Declarar la variable para guardar la imagen*/
    unsigned char *img, *resImg;
    /*Declarar las variables necesarias para leer la imagen*/
    int width = 0, height = 0, channels = 0;
    /*Variable para el número de hilos*/
    int nThreads = 0;
    /*Variable para el número de bloques*/
    int nBlocks = 0;
    /*Crear las tres matrices para cada canal de color*/
    int *matR, *matG, *matB;
    /*Crear las tres matrices del device*/
    int *d_MatR, *d_MatG, *d_MatB;
    /*Crear las tres matrices para cada canal de color*/
    int *rMatR, *rMatG, *rMatB;
    /*Crear las tres matrices resultantes del device*/
    int *d_rMatR, *d_rMatG, *d_rMatB;
    /*Crear variable para el sizeof int*/
    int size = sizeof(int);
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    /* Error code to check return values for CUDA calls */
    cudaError_t err = cudaSuccess;
    /*Creación de matriz con los posibles kernel*/
    int *ker;
    int *d_ker;
    int kernels[6][9] = {
        {-1, 0, 1, -2, 0, 2, -1, 0, 1},     // Border detection (Sobel)
        {1, -2, 1, -2, 5, -2, 1, -2, 1},    // Sharpen
        {1, 1, 1, 1, -2, 1, -1, -1, -1},    // Norte
        {-1, 1, 1, -1, -2, 1, -1, 1, 1},    // Este
        {-1, -1, 0, -1, 0, 1, 0, 1, 1},     // Estampado en relieve
        {-1, -1, -1, -1, 8, -1, -1, -1, -1} // Border detection (Sobel2)
    };
    /*Verificar que la cantidad de argumentos sea la correcta*/
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./my-effect input_image output_image kernel_parameter nThreads\n");
        exit(1);
    }
    /*Cargar en las variables los parametros*/
    loadPath = *(argv + 1);
    savePath = *(argv + 2);
    argKer = atoi(*(argv + 3));
    nBlocks = atoi(*(argv + 4));
    nThreads = atoi(*(argv + 5));
    /*Verificar que el número de hilos sea válido*/
    if (nThreads <= 0 || nBlocks <= 0)
    {
        printf("El número de hilos ingresado o de bloques no es válido \n");
        exit(1);
    }
    if (argKer > 5)
    {
        printf("El parámetro de kernel debe ser menor o igual a 5 \n");
        exit(1);
    }
    /*Cargar en el ker el kernel escogido por el usuario*/
    ker = *(kernels + argKer);
    /*Cargar la imagen usando el parámetro con el nombre*/
    img = stbi_load(loadPath, &width, &height, &channels, 0);
    /*Verificar que la imagen exista y sea cargada correctamente*/
    if (img == NULL)
    {
        printf("Error al cargar la imagen \n");
        exit(1);
    }
    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);
    /*Crear cada matriz de Color dependiendo del tamaño*/
    matR = (int *)malloc(height * width * size);
    matG = (int *)malloc(height * width * size);
    matB = (int *)malloc(height * width * size);
    rMatR = (int *)malloc(height * width * size);
    rMatG = (int *)malloc(height * width * size);
    rMatB = (int *)malloc(height * width * size);
    if (matR == NULL || matG == NULL || matB == NULL || rMatR == NULL || rMatG == NULL || rMatB == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
        exit(1);
    }

    /*Inicializar las matrices con los valores de la imagen*/
    initializeMatrix(matR, matG, matB, width, height, channels, img);

    err = cudaMalloc((void **)&d_ker, size * 9);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_ker (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy inputs to device
    err = cudaMemcpy(d_ker, ker, size * 9, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy ker from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Logica del filtro*/
    /*Crear las matrices de Color con para los resultados*/
    err = cudaMalloc((void **)&d_MatR, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatR (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_MatG, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatG (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_MatB, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatB (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy inputs to device
    err = cudaMemcpy(d_MatR, matR, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatR from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_MatG, matG, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatG from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_MatB, matB, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatB from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /******************************************/

    /*Crear las matrices de Color con para los resultados*/
    err = cudaMalloc((void **)&d_rMatR, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatR (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_rMatG, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatG (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_rMatB, height * width * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device MatB (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Paralelizar el algoritmo*/
    applyFilter<<<nBlocks, nThreads>>>(d_MatR, d_MatG, d_MatB, d_rMatR, d_rMatG, d_rMatB, width, height, nBlocks * nThreads, d_ker);

    cudaDeviceSynchronize();
    
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy result back to host
    err = cudaMemcpy(rMatR, d_rMatR, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rMatR from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(rMatG, d_rMatG, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rMatG from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(rMatB, d_rMatB, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rMatB from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*******************************************/

    /*Exportar la imagen resultante*/
    /*Reservar el espacio de memoria para la imagen resultante*/
    resImg = (unsigned char *)malloc(width * height * channels);
    if (resImg == NULL)
    {
        printf("Error al crear la imagen, problema con malloc \n");
        exit(1);
    }

    joinMatrix(rMatR, rMatG, rMatB, width, height, channels, resImg, img);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);
    /*Guardar la imagen con el nombre especificado*/
    if (strstr(savePath, ".png"))
        stbi_write_png(savePath, width, height, channels, resImg, width * channels);
    else
        stbi_write_jpg(savePath, width, height, channels, resImg, EXPORT_QUALITY);
    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);
    /*Imprimir informe*/
    printf("------------------------------------------------------------------------------\n");
    printf("Número de bloques: %d,  Número de hilos: %d,  Imagen carga: %s\n", nBlocks, nThreads, loadPath);
    printf("Resolución: %dp,  Número de kernel (Parámetro): %d, Imagen exportada: %s\n", height, argKer, savePath);
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("Resumen: (RES, BLOQUES, HILOS, PARAM, TIEMPO) \t%dp\t%d\t%d\t%d\t%ld.%06ld\t\n", height, nBlocks, nThreads, argKer, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    /* Escribir los resultados en un csv*/
    fp = fopen("times.csv", "a");
    if (fp == NULL)
    {
        printf("Error al abrir el archivo \n");
        exit(1);
    }
    fprintf(fp, "%d,%d,%d,%d,%ld.%06ld\n", height, nBlocks, nThreads, argKer, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    fclose(fp);
    /*Liberar memoria*/
    free(matR);
    free(matG);
    free(matB);
    cudaFree(d_MatR);
    cudaFree(d_MatG);
    cudaFree(d_MatB);
    cudaFree(d_rMatR);
    cudaFree(d_rMatG);
    cudaFree(d_rMatB);
    cudaFree(d_ker);
    free(resImg);
    stbi_image_free(img);
    return 0;
}
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

/*Variable para el número de hilos*/
int nThreads = 0;

/*Variable para el número de bloques*/
int nBlocks = 0;

/*Crear las tres matrices para cada canal de color*/
int *matR, *matG, *matB;
/*Crear las tres matrices del device*/
int *d_MatR, *d_MatG, *d_MatB;
/*Crear las tres matrices resultantes del device*/
int *d_rMatR, *d_rMatG, *d_rMatB;

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

__global__ void applyFilter(int *matR, int *matG, int *matB, int *rMatR, int *rMatG, int *rMatB, int width, int height, int nBlocks, int nThreads, int *ker)
{
    __shared__ int *tMatR;
    __shared__ int sizeBlock;
    int i = 0, j = 0;
    int sPosBlock = (blockIdx.x < (width * height) % nBlocks) ? ((width * height) / nBlocks) * blockIdx.x + blockIdx.x : ((width * height) / nBlocks) * blockIdx.x + (width * height) % nBlocks;
    int ePosBlock = (blockIdx.x < (width * height) % nBlocks) ? sPosBlock + ((width * height) / nBlocks) : sPosBlock + ((width * height) / nBlocks) - 1;
    sizeBlock = ePosBlock - sPosBlock + 1;
    tMatR = (int *)malloc((sizeBlock) * sizeof(int));
    if (tMatR == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
    }
    int sPosThr = (threadIdx.x < sizeBlock % nThreads) ? (sizeBlock / nThreads) * threadIdx.x + threadIdx.x : (sizeBlock / nThreads) * threadIdx.x + (sizeBlock) % nThreads;
    int ePosThr = (threadIdx.x < sizeBlock % nThreads) ? sPosThr + (sizeBlock / nThreads) : sPosThr + (sizeBlock / nThreads) - 1;

    printf("%d : %d  -  %d : %d   ___  %d  ***   %d : %d\n", blockIdx.x, threadIdx.x, sPosBlock, ePosBlock, sizeBlock, sPosThr, ePosThr);

    for (i = 0; i <= ePosThr; i++)
        *(tMatR + i) = *(matR + i + sPosBlock);

    /*for (i = 0; i <= ePosThr; i++)
    {
        printf("%d ", *(tMatR + i));
    }*/

    __syncthreads();

    for (i = 0; i <= ePosThr; i++)
    {
        *(rMatR + i + sPosBlock) = *(tMatR + i) - 150;
    }

    /*
    if (tMatR == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
    }

    int sizeThMat = (threadIdx.x < sizeMat % nThreads) ? (sizeMat / nThreads) + 1 : sizeMat / nThreads;

    for (int i = 0; i < sizeThMat; i++)
    {
        *(rMatR + i) = 80;
    }
    printf("%d : %d  -  %d \n", blockIdx.x, threadIdx.x, sizeThMat);
    */

    // int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
    // int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
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
    /*Crear variable para el sizeof int*/
    int size = sizeof(int);
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    /* Error code to check return values for CUDA calls */
    cudaError_t err = cudaSuccess;
    /*Creación de matriz con los posibles kernel*/
    int *ker;
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
    if (matR == NULL || matG == NULL || matB == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
        exit(1);
    }

    /*Inicializar las matrices con los valores de la imagen*/
    initializeMatrix(matR, matG, matB, width, height, channels, img);

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

    // Copy inputs to device
    err = cudaMemcpy(d_rMatR, matR, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatR from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_rMatG, matG, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatG from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_rMatB, matB, height * width * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatB from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Paralelizar el algoritmo*/
    applyFilter<<<nBlocks, nThreads>>>(d_MatR, d_MatG, d_MatB, d_rMatR, d_rMatG, d_rMatB, width, height, nBlocks, nThreads, ker);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy result back to host
    err = cudaMemcpy(matR, d_rMatR, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatR from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(matG, d_rMatG, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatG from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(matB, d_rMatB, height * width * size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy MatB from device to host (error code %s)!\n", cudaGetErrorString(err));
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

    joinMatrix(matR, matG, matB, width, height, channels, resImg, img);

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
    /*
    printf("------------------------------------------------------------------------------\n");
    printf("Número de hilos: %d,  Imagen carga: %s,   Imagen exportada: %s\n", nThreads, loadPath, savePath);
    printf("Resolución: %dp,  Número de kernel (Parámetro): %d\n", height, argKer);
    printf("Imagen exportada: %s\n", savePath);
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("Resumen: (RES, HILOS, PARAM, TIEMPO) \t%dp\t%d\t%d\t%ld.%06ld\t\n", height, nThreads, argKer, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    */
    /* Escribir los resultados en un csv*/
    fp = fopen("times.csv", "a");
    if (fp == NULL)
    {
        printf("Error al abrir el archivo \n");
        exit(1);
    }
    fprintf(fp, "%d,%d,%d,%ld.%06ld\n", height, nThreads, argKer, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    fclose(fp);
    /*Liberar memoria*/
    free(matR);
    free(matG);
    free(matB);
    cudaFree(d_MatR);
    cudaFree(d_MatG);
    cudaFree(d_MatB);
    free(resImg);
    stbi_image_free(img);
    return 0;
}
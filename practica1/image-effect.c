/**
 * @file main.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-04-05
 * @copyright Copyright (c) 2022
 */

/*Para la lectura y escritura de imagenes se usan las librerias stb_image*/
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define R_ARGS 4
#define EXPORT_QUALITY 100

/*Crear las tres matrices para cada canal de color*/
int *matR, *matG, *matB;
/*Crear las tres matrices resultanres*/
int *rMatR, *rMatG, *rMatB;

/*Declarar las variables necesarias para leer la imagen*/
int width = 0, height = 0, channels = 0;

/*Variable para el número de hilos*/
int nThreads = 0;

int *ker;

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

void *applyFilter(void *arg)
{
    /*Variables necesarias para la convolución*/
    int thread_id = *(int *)arg;
    int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
    int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
    int conv = 0;
    /*Calcular la posición inicial en términos de i y j*/
    int i = (startPos / width), j = (startPos % width);
    /*Realizar la convolucion*/
    for (startPos; startPos < endPos; startPos++)
    {
        /*Ignorar la convolucion en los bordes*/
        if (i > 0 && i < height - 1 && j > 0 && j < width - 1)
        {
            /*Convolucion para el canal R*/
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
    /**/
    int i;
    /*Creación de matriz con los posibles kernel*/
    int kernels[6][9] = {
        {-1, 0, 1, -2, 0, 2, -1, 0, 1},     // Border detection (Sobel)
        {1, -2, 1, -2, 5, -2, 1, -2, 1},    // Sharpen
        {1, 1, 1, 1, -2, 1, -1, -1, -1},    // Norte
        {-1, 1, 1, -1, -2, 1, -1, 1, 1},    // Este
        {-1, -1, 0, -1, 0, 1, 0, 1, 1},     // Estampado en relieve
        {-1, -1, -1, -1, 8, -1, -1, -1, -1} // Border detection (Sobel2)
    };
    /*Verificar que la cantidad de argumentos sea la correcta*/
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./main input_image output_image parameter threads\n");
        exit(1);
    }
    /*Cargar en las variables los parametros*/
    loadPath = *(argv + 1);
    savePath = *(argv + 2);
    argKer = atoi(*(argv + 3));
    nThreads = atoi(*(argv + 4));
    /*Verificar que el número de hilos sea válido*/
    if (nThreads < 0)
    {
        printf("El número de hilos ingresado no es válido \n");
        exit(1);
    }
    if (argKer > 5)
    {
        printf("El parámetro de kernel debe ser menor o igual a 5 \n");
        exit(1);
    }
    /*Declaración de variables de paralelización*/
    int threadId[nThreads];
    pthread_t thread[nThreads];
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
    /*Crear cada matriz de Color dependiendo del tamaño*/
    matR = (int *)malloc(height * width * sizeof(int));
    matG = (int *)malloc(height * width * sizeof(int));
    matB = (int *)malloc(height * width * sizeof(int));
    if (matR == NULL || matG == NULL || matB == NULL)
    {
        printf("Error al crear las matrices, problema con malloc \n");
        exit(1);
    }
    /*Inicializar las matrices con los valores de la imagen*/
    initializeMatrix(matR, matG, matB, width, height, channels, img);
    /*Logica del filtro*/
    /*Crear las matrices de Color con para los resultados*/
    rMatR = (int *)calloc(height * width, sizeof(int));
    rMatG = (int *)calloc(height * width, sizeof(int));
    rMatB = (int *)calloc(height * width, sizeof(int));
    /*Paralelizar el algoritmo*/
    for (i = 0; i < nThreads; i++)
    {
        threadId[i] = i;
        pthread_create(&thread[i], NULL, (void *)applyFilter, (void *)&threadId[i]);
    }

    for (i = 0; i < nThreads; i++)
    {
        pthread_join(thread[i], NULL);
    }
    /*Exportar la imagen resultante*/
    /*Reservar el espacio de memoria para la imagen resultante*/
    resImg = malloc(width * height * channels);
    if (resImg == NULL)
    {
        printf("Error al crear la imagen, problema con malloc \n");
        exit(1);
    }
    joinMatrix(rMatR, rMatG, rMatB, width, height, channels, resImg, img);
    /*Guardar la imagen con el nombre especificado*/
    if (strstr(savePath, ".png"))
        stbi_write_png(savePath, width, height, channels, resImg, width * channels);
    else
        stbi_write_jpg(savePath, width, height, channels, resImg, EXPORT_QUALITY);
    /*Liberar memoria*/
    free(matR);
    free(matG);
    free(matB);
    free(rMatR);
    free(rMatG);
    free(rMatB);
    free(resImg);
    stbi_image_free(img);
    return 0;
}
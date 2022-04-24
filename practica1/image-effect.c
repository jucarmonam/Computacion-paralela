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

int main(int argc, char *argv[])
{
    /*Declarar los string de lectura y escritura*/
    char *loadPath, *savePath;
    /*Variable para yo que se xd*/
    int arg = 0;
    /*Variable para el número de hilos*/
    int nThreads = 0;
    /*Declaración de variable para la escritura del archivo*/
    FILE *fp;
    /*Declarar las variables necesarias para leer la imagen*/
    int width = 0, height = 0, channels = 0;
    /*Declarar la variable para guardar la imagen*/
    unsigned char *img, *resImg;
    /*Crear las tres matrices para cada canal de color*/
    int *matR, *matG, *matB;
    /*Crear las tres matrices resultanres*/
    int *rMatR, *rMatG, *rMatB;
    /**/
    int kernel[] = {-1,0,1,-2,0,2,-1,0,1};
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
    arg = atoi(*(argv + 3));
    nThreads = atoi(*(argv + 4));
    /*Verificar que el número de hilos sea válido*/
    if (nThreads < 0)
    {
        printf("El número de hilos ingresado no es válido \n");
        exit(1);
    }
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
    rMatR = (int *)malloc(height * width * sizeof(int));
    rMatG = (int *)malloc(height * width * sizeof(int));
    rMatB = (int *)malloc(height * width * sizeof(int));

    int i = 0;
    int j = 0;
    int conv = 0;
    for (i = 1; i < height-1; i++)
    {
        for (j = 1; j < width-1; j++)
        {



            
            
            

            /*Convolucion para el canal R*/
            conv = (*(kernel) * *(matR + ((i - 1) * width + j - 1)) + *(kernel+1) * *(matR + ((i - 1) * width + j)) + *(kernel + 2) * *(matR + ((i - 1) * width + j + 1))
                                        + *(kernel+3) * *(matR + (i  * width + j - 1)) + *(kernel+4) * *(matR + (i * width + j)) + *(kernel+ 5) * *(matR + (i * width + j + 1))
                                        + *(kernel+6) * *(matR + ((i + 1) * width + j - 1)) + *(kernel+7) * *(matR + ((i + 1) * width + j)) + *(kernel+8) * *(matR + ((i + 1) * width + j + 1)))%255;
            
            *(rMatR + (i * width + j)) = conv < 0 ? 0 : conv;
            /*Convolucion para el canal G*/

            
            conv = (*(kernel) * *(matG + ((i - 1) * width + j - 1)) + *(kernel+1) * *(matG + ((i - 1) * width + j)) + *(kernel + 2) * *(matG + ((i - 1) * width + j + 1))
                                        + *(kernel+3) * *(matG + (i  * width + j - 1)) + *(kernel+4) * *(matG + (i * width + j)) + *(kernel+ 5) * *(matG + (i * width + j + 1))
                                        + *(kernel+6) * *(matG + ((i + 1) * width + j - 1)) + *(kernel+7) * *(matG + ((i + 1) * width + j)) + *(kernel+8) * *(matG + ((i + 1) * width + j + 1)))%255;
            *(rMatG + (i * width + j)) = conv < 0 ? 0 : conv;
            /*Convolucion para el canal B*/

            //*(rMatB + (i * width + j)) = 0;

            
            conv = (*(kernel) * *(matB + ((i - 1) * width + j - 1)) + *(kernel+1) * *(matB + ((i - 1) * width + j)) + *(kernel + 2) * *(matB + ((i - 1) * width + j + 1))
                                        + *(kernel+3) * *(matB + (i  * width + j - 1)) + *(kernel+4) * *(matB + (i * width + j)) + *(kernel+ 5) * *(matB + (i * width + j + 1))
                                        + *(kernel+6) * *(matB + ((i + 1) * width + j - 1)) + *(kernel+7) * *(matB + ((i + 1) * width + j)) + *(kernel+8) * *(matB + ((i + 1) * width + j + 1)))%255;
            *(rMatB + (i * width + j)) = conv < 0 ? 0 : conv;

        }
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
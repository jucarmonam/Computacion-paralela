/**
 * @file matMult.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-0-06
 * @copyright Copyright (c) 2022
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <sys/time.h>

#define R_ARGS 4

/*Variable para el size del dato*/
int size = sizeof(int);

void readMatrix(char *path, char *matrixS, int n)
{
    FILE *fp;
    int i = 0;
    char c;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        printf("Error el abrir el archivo...\n");
        exit(1);
    }
    do
    {

        c = fgetc(fp);
        *(matrixS + i) = c;
        i++;
        if (feof(fp))
            break;
    } while (1);

    fclose(fp);
}

void fillMatrix(char *matS, int *matrix, int n)
{
    int i = 0;
    char *delimiter = "_";
    char *token = strtok(matS, delimiter);
    while (token != NULL)
    {
        if (i >= n * n)
        {
            printf("Las dimensiones de la matriz no coinciden con la matriz ingresada. (Más elementos) \n");
            exit(1);
        }
        *(matrix + i) = atoi(token);
        token = strtok(NULL, delimiter);
        i++;
    }
    if (i < (n * n))
    {
        printf("Las dimensiones de la matriz no coinciden con la matriz ingresada. (Menos elementos) \n");
        exit(1);
    }
}

void transpose(int *matrix, int *matrixT, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(matrixT + j * n + i) = *(matrix + i * n + j);
        }
    }
}

void transposeP(int *matrix, int *matrixT, int n, int nThreads)
{
    int matrixSize = n * n;

#pragma omp parallel num_threads(nThreads)
    {
        int thread_id = omp_get_thread_num(), iter;
        int start = (thread_id < matrixSize % nThreads) ? (matrixSize / nThreads) * thread_id + thread_id : (matrixSize / nThreads) * thread_id + matrixSize % nThreads;
        int end = (thread_id < matrixSize % nThreads) ? start + (matrixSize / nThreads) : start + (matrixSize / nThreads) - 1;
        int i = (start / n), j = (start % n);

        for (iter = start; iter <= end; iter++)
        {
            *(matrixT + j * n + i) = *(matrix + i * n + j);
            j += 1;
            if (j == n)
            {
                i += 1;
                j = 0;
            }
        }
    }
}

void matMult(int *A, int *B, int *C, int n, int nThreads)
{
    int i, j, k;
    int *BT = malloc(n * n * size);
    transpose(B, BT, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            int mult = 0;
            for (k = 0; k < n; k++)
            {
                mult += *(A + i * n + k) * *(BT + j * n + k);
            }
            *(C + i * n + j) = mult;
        }
    }
    free(BT);
}

void matMultP(int *A, int *B, int *C, int n, int nThreads)
{
    int matrixSize = n * n;
    int *BT = malloc(n * n * size);
    transposeP(B, BT, n, 16);
#pragma omp parallel num_threads(nThreads)
    {
        int thread_id = omp_get_thread_num(), iter;
        int start = (thread_id < matrixSize % nThreads) ? (matrixSize / nThreads) * thread_id + thread_id : (matrixSize / nThreads) * thread_id + matrixSize % nThreads;
        int end = (thread_id < matrixSize % nThreads) ? start + (matrixSize / nThreads) : start + (matrixSize / nThreads) - 1;
        int i = (start / n), j = (start % n), k;
        for (iter = start; iter <= end; iter++)
        {
            int mult = 0;
            for (k = 0; k < n; k++)
            {
                mult += *(A + i * n + k) * *(BT + j * n + k);
            }
            *(C + i * n + j) = mult;
            j += 1;
            if (j == n)
            {
                i += 1;
                j = 0;
            }
        }
    }
    free(BT);
}

int main(int argc, char *argv[])
{
    /*Arreglo para la mariz A y B*/
    int *A, *B, *C, *C1;
    /*Variables para el PATH de matA y matB*/
    char *pathA, *pathB;
    /*Variable para la matriz A como String*/
    char *matrixAS;
    /*Variable para la matriz B como String*/
    char *matrixBS;
    /*Variable para la dimensión N de la matriz y el tamaño máximo de la matriz*/
    int n, maxSize;
    /*Variable para el número de hilos*/
    int nThreads = 0;
    /*Variable para el tamaño del número */
    int numSize;
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    /*Verificar que el número de argumentos sea correcto*/
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./matMult matrixA matrixB n nThreads\n");
        exit(1);
    }

    /*Cargar en las variables los parametros*/
    pathA = *(argv + 1);
    pathB = *(argv + 2);
    n = atoi(*(argv + 3));
    numSize = (int)strlen(*(argv + 3));
    nThreads = atoi(*(argv + 4));
    /*Verificar que el número de hilos sea válido*/
    if (nThreads < 0)
    {
        printf("El número de hilos ingresado no es válido \n");
        exit(1);
    }
    if (n < 2)
    {
        printf("La matriz debe ser de tamaño mayor o igual a 2 * 2 \n");
        exit(1);
    }
    /*Definir el tamaño maximo dependiendo del N de la matriz*/
    maxSize = n * n * (numSize + 1 + 1);
    /*Reservar el espacio para las matrices*/
    matrixAS = malloc(maxSize);
    matrixBS = malloc(maxSize);
    /*Cargar las matrices del PATH*/
    readMatrix(pathA, matrixAS, n);
    readMatrix(pathB, matrixBS, n);
    /*Reservar en memoria el espacio para las matrices*/
    A = (int *)malloc(n * n * size);
    B = (int *)malloc(n * n * size);
    C = (int *)malloc(n * n * size);
    C1 = (int *)malloc(n * n * size);

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Error al crear las matrices, error en la ejecución de malloc \n");
        exit(1);
    }
    /*Se llenan las matrices con los valores almacenados en los String*/
    fillMatrix(matrixAS, A, n);
    fillMatrix(matrixBS, B, n);

    /*
    for (int i = 0; i < n * n; i++)
    {
        if (i % n == 0)
            printf("\n");
        printf("%d ", *(A + i));
    }
    printf("\n");
    printf("\n");

    for (int i = 0; i < n * n; i++)
    {
        if (i % n == 0)
            printf("\n");
        printf("%d ", *(B + i));
    }
    printf("\n");
    */
    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    matMult(A, B, C, n, nThreads);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);

    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);

    printf("Tiempo de ejecución Secuencial: %ld.%010ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    matMultP(A, B, C1, n, nThreads);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);

    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);

    printf("Tiempo de ejecución Paralelo: %ld.%010ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    for (int i = 0; i < n * n; i++)
    {
        if (*(C + i) != *(C + i))
        {
            printf("Resultados diferentes \n");
            exit(1);
        }
    }
    printf("Multiplicación realizada satisfactoriamente :)\n");

    /*Liberar memoria*/
    free(A);
    free(B);
    free(C);
    free(C1);

    return 0;
}

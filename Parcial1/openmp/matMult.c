/**
 * @file matMult.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-0-06
 * @copyright Copyright (c) 2022
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <sys/time.h>

void transpose(int *A, int *AT, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(AT + i * n + j) = *(A + j * n + i);
        }
    }
}

void transposePar(int *A, int *AT, int n)
{
    int matrixSize = n * n;
#pragma omp parallel
    {
        /*Obtener el id del hilo*/
        int id = omp_get_thread_num();
        int nThreads = omp_get_num_threads();
        int start = (id < matrixSize % nThreads) ? (matrixSize / nThreads) * id + id : (matrixSize / nThreads) * id + matrixSize % nThreads;
        int end = (id < matrixSize % nThreads) ? start + (matrixSize / nThreads) : start + (matrixSize / nThreads) - 1;

        int i = (start / n), j = (start % n);

        int iter;

        for (iter = start; iter <= end; iter++)
        {
            *(AT + i * n + j) = *(A + j * n + i);
            j += 1;
            if (j == n)
            {
                i += 1;
                j = 0;
            }
        }
    }
}

void gemmT(int *A, int *B, int *C, int n)
{
    int i, j, k;
    int *B2;
    B2 = (int *)malloc(sizeof(int) * n * n);
    transpose(B, B2, n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            int dot = 0;
            for (k = 0; k < n; k++)
            {
                dot += A[i * n + k] * B2[j * n + k];
            }
            C[i * n + j] = dot;
        }
    }
    free(B2);
}

void matMultPar(int *A, int *B, int *C, int n, int nThreads)
{
    int *B2;
    int matrixSize = n * n;
    B2 = (int *)malloc(sizeof(int) * n * n);
    transposePar(B, B2, n);
#pragma omp parallel num_threads(nThreads)
    {
        int id = omp_get_thread_num();
        int start = (id < matrixSize % nThreads) ? (matrixSize / nThreads) * id + id : (matrixSize / nThreads) * id + matrixSize % nThreads;
        int end = (id < matrixSize % nThreads) ? start + (matrixSize / nThreads) : start + (matrixSize / nThreads) - 1;

        int i = (start / n), j = (start % n), k;

        int iter;

        for (iter = start; iter <= end; iter++)
        {
            int dot = 0;
            for (k = 0; k < n; k++)
            {
                dot += *(A + i * n + k) * *(B2 + j * n + k);
            }
            *(C + i * n + j) = dot;
            j += 1;
            if (j == n)
            {
                i += 1;
                j = 0;
            }
        }
    }

    free(B2);
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    int i, j, n = 5000;
    int nThreads = 32;
    int *A, *B, *C, *C1;
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    A = (int *)malloc(sizeof(int) * n * n);
    B = (int *)malloc(sizeof(int) * n * n);
    C = (int *)malloc(sizeof(int) * n * n);
    C1 = (int *)malloc(sizeof(int) * n * n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(A + i * n + j) = (rand() % 40) - 20;
            *(B + i * n + j) = (rand() % 40) - 20;
        }
    }
    /*
    printf("A: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %d ", *(A + i * n + j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");

    printf("B: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %d ", *(B + i * n + j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    */
    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    gemmT(A, B, C, n);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);
    /*
    printf("---------------------------------------\n");
    printf("C: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %d ", *(C + i * n + j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    */
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\n Tiempo de ejecución Secuencial:  %ld.%015ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("---------------------------------------\n");

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    matMultPar(A, B, C1, n, 16);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);
    /*
    printf("---------------------------------------\n");
    printf("C: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %d ", *(C + i * n + j));
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    */
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\n Tiempo de ejecución Paralelo:  %ld.%015ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("---------------------------------------\n");

    for (i = 0; i < n; i++)
    {
        if (*(C + i) != *(C1 + i))
        {
            printf("Resultados diferentes \n");
            exit(-1);
        }
    }
    printf("Resultados Iguales :) \n");

    free(A);
    free(B);
    free(C);
    free(C1);

    return 0;
}

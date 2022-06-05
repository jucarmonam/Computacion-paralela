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

void transpose(double *A, double *AT, int n)
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

void transposePar(double *A, double *AT, int n, int nThreads)
{
    int matrixSize = n * n;
#pragma omp parallel num_threads(nThreads)
    {
        /*Obtener el id del hilo*/
        int id = omp_get_thread_num();
        int start = (id < matrixSize % nThreads) ? (matrixSize / nThreads) * id + id : (matrixSize / nThreads) * id + matrixSize % nThreads;
        int end = (id < matrixSize % nThreads) ? start + (matrixSize / nThreads) : start + (matrixSize / nThreads) - 1;

        // printf(" id: %d   -   start: %d    end: %d \n", id, start, end);

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
        // printf("\n \n");
    }
}

void gemmT(double *A, double *B, double *C, int n)
{
    int i, j, k;
    double *B2;
    B2 = (double *)malloc(sizeof(double) * n * n);
    transpose(B, B2, n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            double dot = 0;
            for (k = 0; k < n; k++)
            {
                dot += A[i * n + k] * B2[j * n + k];
            }
            C[i * n + j] = dot;
        }
    }
    free(B2);
}

int main(int argc, char const *argv[])
{
    srand(time(NULL)); 
    int i, j, n = 3000;
    int nThreads = 32;
    double *A, *AT;
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    A = (double *)malloc(sizeof(double) * n * n);
    AT = (double *)malloc(sizeof(double) * n * n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            *(A + i * n + j) = i * n + j;
        }
    }
    /*
    printf("Original \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %f ", *(A + i * n + j));
        }
        printf("\n");
    }*/
    printf("\n");
    printf("\n");

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    transpose(A, AT, n);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);



    /*
    printf("Transposed matrix: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %f ", *(AT + i * n + j));
        }
        printf("\n");
    }
    */
    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\nTiempo de ejecución Secuencial: %ld.%015ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    printf("--------------------------------\n");


    free(AT);
    AT = (double *)malloc(sizeof(double) * n * n);

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    transposePar(A, AT, n, nThreads);

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);
    /*
    printf("Transposed matrix Parallel: \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf(" %f ", *(AT + i * n + j));
        }
        printf("\n");
    }
    */
    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\n Tiempo de ejecución Paralelo:  %ld.%015ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    

    /*Parallel test*/
    free(A);
    free(AT);
    return 0;
}

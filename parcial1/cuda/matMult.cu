/**
 * @file matMult.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-0-06
 * @copyright Copyright (c) 2022
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define R_ARGS 6

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
    char *delimiter = (char *)"_";
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

void matMult(int *A, int *B, int *C, int n)
{
    int i, j, k;
    int *BT = (int *)malloc(n * n * size);
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

__global__ void transposeP(int *matrix, int *matrixT, int n, int nThreads)
{
    int matrixSize = n * n;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int iter;

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

__global__ void matMultP(int *A, int *BT, int *C, int n, int nThreads)
{
    int matrixSize = n * n;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x, iter;

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

void checkMatrix(int *matrix, int *checkMatrix, int n)
{
    int i;
    for (i = 0; i < n * n; i++)
    {
        if (*(matrix + i) != *(checkMatrix + i))
        {
            printf("Los resultados calculados en la multiplicación son diferentes \n");
            exit(1);
        }
    }
}

void writeResult(int *matrix, int n)
{
    /*Declaración de variable para la escritura del archivo*/
    FILE *fp;
    int i;
    fp = fopen("../files/matResCUDA.txt", "w+");
    if (fp == NULL)
    {
        printf("Error al leer el archivo files/matResCUDA.txt\n");
        exit(1);
    }
    for (i = 0; i < n * n; i++)
    {
        fprintf(fp, "%d", *(matrix + i));
        if (i < n * n - 1)
            fprintf(fp, "_");
    }
}

int main(int argc, char *argv[])
{
    /*Declaración de variable para la escritura del archivo*/
    FILE *fp;
    /*Variables i, j*/
    int i, j;
    /*Arreglo para la mariz A, B y C*/
    int *A, *B, *C, *BT, *ChkRes;
    /*Crear las matrices en el device*/
    int *d_A, *d_B, *d_C, *d_BT;
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
    /*Variable para el número de bloques*/
    int nBlocks = 0;
    /*Variable para el número de cifras del N */
    int numSize;
    /*Variable de automatización*/
    int test;
    /* Error code to check return values for CUDA calls */
    cudaError_t err = cudaSuccess;
    /*Variables necesarias para medir tiempos*/
    struct timeval tval_before, tval_after, tval_result;
    /*Verificar que el número de argumentos sea correcto*/
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./matMult pathMatrixA pathMatrixB n nBlocks nThreads testing\n");
        exit(1);
    }

    /*Cargar en las variables los parametros*/
    pathA = *(argv + 1);
    pathB = *(argv + 2);
    n = atoi(*(argv + 3));
    numSize = (int)strlen(*(argv + 3));
    nBlocks = atoi(*(argv + 4));
    nThreads = atoi(*(argv + 5));
    test = atoi(*(argv + 6));
    /*Verificar que el número de hilos y bloques sea válido*/
    if (nThreads <= 0 || nBlocks <= 0)
    {
        printf("El número de hilos ingresado o de bloques no es válido \n");
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
    matrixAS = (char *)malloc(maxSize);
    matrixBS = (char *)malloc(maxSize);

    /*Cargar las matrices del PATH*/
    readMatrix(pathA, matrixAS, n);
    readMatrix(pathB, matrixBS, n);

    /*Reservar en memoria el espacio para las matrices*/
    A = (int *)malloc(n * n * size);
    B = (int *)malloc(n * n * size);
    BT = (int *)malloc(n * n * size);
    C = (int *)malloc(n * n * size);
    ChkRes = (int *)malloc(n * n * size);

    if (A == NULL || B == NULL || C == NULL || BT == NULL || ChkRes == NULL)
    {
        printf("Error al crear las matrices, error en la ejecución de malloc \n");
        exit(1);
    }

    /*Se llenan las matrices con los valores almacenados en los String*/
    fillMatrix(matrixAS, A, n);
    fillMatrix(matrixBS, B, n);

    /*Realizar la multiplicación con el algoritmo secuencial para verificar*/
    matMult(A, B, ChkRes, n);

    /* Escribir los resultados en un csv*/
    fp = fopen("../files/timesCUDA.csv", "a");
    if (fp == NULL)
    {
        printf("Error al abrir el archivo csv \n");
        exit(1);
    }

    printf("------------------------------------------------\n");
    printf("                      CUDA                      \n");
    printf("------------------------------------------------\n");
    printf("              Matrix  %d x %d                   \n", n, n);
    printf("------------------------------------------------\n");

    /*Reservar en memoria una copia de la matriz A*/
    err = cudaMalloc((void **)&d_A, size * n * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Copiar la matriz A del Host al Device*/
    err = cudaMemcpy(d_A, A, size * n * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Reservar en memoria una copia de la matrix B*/
    err = cudaMalloc((void **)&d_B, size * n * n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Copiar matrix B del Host al Device*/
    err = cudaMemcpy(d_B, B, size * n * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Reservar espacio en memoria para la matriz transpuesta de B en device*/
    err = cudaMalloc((void **)&d_BT, n * n * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_BT (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Reservar espacio en memoria para la matriz de resultados*/
    err = cudaMalloc((void **)&d_C, n * n * size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*En caso de que el parametro de test sea 0, se realiza una única prueba*/
    if (test == 0)
    {
        /*Medición de tiempo de inicio*/
        gettimeofday(&tval_before, NULL);

        /*Realizar la transposición*/
        transposeP<<<nBlocks, nThreads>>>(d_B, d_BT, n, nBlocks * nThreads);

        cudaDeviceSynchronize();

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch transposeP (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        /*Realizar la multiplicación de matrices*/
        matMultP<<<nBlocks, nThreads>>>(d_A, d_BT, d_C, n, nBlocks * nThreads);

        cudaDeviceSynchronize();

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch matMultP (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        /*Medición de tiempo de finalización*/
        gettimeofday(&tval_after, NULL);

        /* Copiar el resultado de vuelta al host*/
        err = cudaMemcpy(C, d_C, size * n * n, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy C from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        checkMatrix(C, ChkRes, n);

        /*Calcular los tiempos en tval_result*/
        timersub(&tval_after, &tval_before, &tval_result);

        printf("Tiempo de ejecución ( %d bloques, %d hilos ): %ld.%06ld s \n", nBlocks, nThreads, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
        fprintf(fp, "%d,%d,%d,%ld.%06ld\n", n, nBlocks, nThreads, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    }
    else
    {
        for (i = 1; i <= nBlocks; i *= 2)
        {
            printf("------------------------------------------------\n");
            for (j = 1; j <= nThreads; j += 20)
            {
                /*Medición de tiempo de inicio*/
                gettimeofday(&tval_before, NULL);

                /*Paralelizar el algoritmo*/
                transposeP<<<i, j>>>(d_B, d_BT, n, i * j);

                cudaDeviceSynchronize();

                err = cudaGetLastError();

                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch transposeP (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }

                matMultP<<<i, j>>>(d_A, d_BT, d_C, n, i * j);

                cudaDeviceSynchronize();

                err = cudaGetLastError();

                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch matMultP (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }

                /*Medición de tiempo de finalización*/
                gettimeofday(&tval_after, NULL);

                /*Calcular los tiempos en tval_result*/
                timersub(&tval_after, &tval_before, &tval_result);

                printf("Tiempo de ejecución ( %d bloques , %d hilos ): %ld.%06ld s \n", i, j, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
                fprintf(fp, "%d,%d,%d,%ld.%06ld\n", n, i, j, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

                // Copiar el resultado de vuelta en el host
                err = cudaMemcpy(C, d_C, size * n * n, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to copy C from device to host (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                checkMatrix(C, ChkRes, n);
                /*Probar caso base*/
                if (j == 1)
                    j = -10;
            }
        }
        i /= 2;
        if (nBlocks > i)
        {
            i = nBlocks;
            printf("------------------------------------------------\n");
            for (j = 10; j <= nThreads; j += 20)
            {
                /*Medición de tiempo de inicio*/
                gettimeofday(&tval_before, NULL);

                /*Paralelizar el algoritmo*/
                transposeP<<<i, j>>>(d_B, d_BT, n, i * j);

                cudaDeviceSynchronize();

                err = cudaGetLastError();

                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch transposeP (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }

                matMultP<<<i, j>>>(d_A, d_BT, d_C, n, i * j);

                cudaDeviceSynchronize();

                err = cudaGetLastError();

                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch matMultP (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }

                /*Medición de tiempo de finalización*/
                gettimeofday(&tval_after, NULL);

                /*Calcular los tiempos en tval_result*/
                timersub(&tval_after, &tval_before, &tval_result);

                printf("Tiempo de ejecución ( %d bloques , %d hilos ): %ld.%06ld s \n", i, j, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
                fprintf(fp, "%d,%d,%d,%ld.%06ld\n", n, i, j, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

                // Copiar el resultado de vuelta en el host
                err = cudaMemcpy(C, d_C, size * n * n, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to copy C from device to host (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                checkMatrix(C, ChkRes, n);
            }
        }
    }
    fclose(fp);
    writeResult(C, n);
    /*Liberar memoria*/
    free(A);
    free(B);
    free(C);
    free(BT);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_BT);
    return 0;
}
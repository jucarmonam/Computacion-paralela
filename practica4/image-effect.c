/**
 * @file main.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-02-06
 * @copyright Copyright (c) 2022
 */
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <math.h>

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

    /*Variable para iteración*/
    int i;

    /*Variable para iteración*/
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
    if ((argc - 1) < R_ARGS){
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./my-effect input_image output_image kernel_parameter nThreads\n");
        exit(1);
    }

    /*Cargar en las variables los parametros*/
    loadPath = *(argv + 1);
    savePath = *(argv + 2);
    argKer = atoi(*(argv + 3));
    nThreads = atoi(*(argv + 4));

    /*Verificar que el número de hilos sea válido*/
    if (nThreads < 0){
        printf("El número de hilos ingresado no es válido \n");
        exit(1);
    }

    if (argKer > 5){
        printf("El parámetro de kernel debe ser menor o igual a 5 \n");
        exit(1);
    }

    /*Cargar en el ker el kernel escogido por el usuario*/
    ker = *(kernels + argKer);

    /*Cargar la imagen usando el parámetro con el nombre*/
    img = stbi_load(loadPath, &width, &height, &channels, 0);

    /*Verificar que la imagen exista y sea cargada correctamente*/
    if (img == NULL){
        printf("Error al cargar la imagen \n");
        exit(1);
    }

    int done = 0, n, processId, numprocs, I, rc, i;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    if (processId == 0) printf("\nLaunching with %i processes", numprocs);
    global_pi = 0.0;

    #pragma omp parallel num_threads(4)
    {
        int threadId = omp_get_thread_num();
        int threadsTotal = omp_get_num_threads();
        int globalId = (processId * threadsTotal) + threadId;
        calculatePi(&local_pi[threadId], globalId, threadsTotal*numprocs);
        #pragma omp single
        {
        for(i = 0; i < threadsTotal; i++)
            global_pi = global_pi + local_pi[i];
        }
        printf("%i ", globalId); fflush(stdout);
    }

    MPI_Reduce(local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (processId == 0) printf("\npi is approximately %.16f, Error is %.16f\n", global_pi, fabs(global_pi - PI25DT));
	MPI_Finalize();
	return 0;
}
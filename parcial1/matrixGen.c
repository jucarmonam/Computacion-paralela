/**
 * @file matMult.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-0-06
 * @copyright Copyright (c) 2022
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define R_ARGS 1

/*Variable para el size del dato*/
int size = sizeof(int);

int main(int argc, char *argv[])
{
    /*Inicializar la semilla de aleatorios*/
    srand(time(NULL)); 
    /*Variable de iteración i y para el tamaño n*/
    int i, n, random;
    /*Verificar que el número de argumentos sea correcto*/
    if ((argc - 1) != R_ARGS)
    {
        printf("Son necesarios %d argumentos para el funcionamiento\n", R_ARGS);
        printf("Para una correcta ejecución: ./test n \n");
        exit(1);
    }

    /*Cargar en las variables los parametros*/
    n = atoi(*(argv + 1));

    if (n < 2)
    {
        printf("La matriz debe ser de tamaño mayor o igual a 2 * 2 \n");
        exit(1);
    }

    for (i = 0; i < n * n; i++)
    {
        random = rand() % (2 * n) - n;
        printf("%d", random);
        if(i < n * n - 1)
            printf("_");
    }

    return 0;
}

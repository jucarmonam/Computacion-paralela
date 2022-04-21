/**
 * @file main.c
 * @author Juan Pablo Carmona Muñoz (jucarmonam) - Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-04-05
 * @copyright Copyright (c) 2022
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define R_ARGS 3

int main(int argc, char *argv[])
{
    char *loadPath;
    char *savePath;
    int arg = 0;
    FILE *fp;
    if ((argc - 1) < R_ARGS)
    {
        printf("Son necesarios 3 argumentos para el funcionamiento\n");
        printf("Para una correcta ejecución: ./main input_image output_image parameter\n");
        exit(-1);
    }
    loadPath = *(argv+1);
    savePath = *(argv+2);
    arg = atoi(*(argv+3));
    printf("%d\n",arg);
    return 0;
}
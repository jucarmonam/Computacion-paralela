/**
 * @file main.c
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-04-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/wait.h>

int Nthreads;
double* sump;
struct timeval start, end;
double StopWatch;

int main(int argc, char* argv[]){

    gettimeofday(&start, NULL); 

    if (argc > 3){
        Nthreads = atoi(argv[1]);
    }else{
        printf("\nUso: %s Hilos Millones-de-Iteraciones\n",argv[0]);
        exit(1);
    }


    gettimeofday(&end, NULL);               
    StopWatch = (double)(end.tv_sec + (double)end.tv_usec/1000000) -
                (double)(start.tv_sec + (double)start.tv_usec/1000000); // Resta los dos tiempos obtenidos.
    printf("The excecution of the program has taken: %.16g miliseconds.\n",StopWatch*1000.0); // Imprime el resultado.
    
    return 0;
}
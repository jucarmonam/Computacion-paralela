#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void readMatrix(char *path, char *matrixS, int n)
{
    FILE *fp;
    int i = 0;
    char c;
    fp = fopen(path, "r");
    if (fp == NULL)
    {
        perror("Error el abrir el archivo...\n");
    }
    do
    {
        *(matrixS + i) = fgetc(fp);
        i++;
        if (feof(fp))
            break;
    } while (1);

    fclose(fp);
}

int main(int argc, char *argv[])
{
    /*Variables para el PATH de matA y matB*/
    char *pathA;
    /*Variable para la matriz A como String*/
    char *matrixAS;
    int n;
    int maxSize;
    pathA = *(argv + 1);
    n = atoi(*(argv + 2));

    maxSize = n * n * (6);
    matrixAS = malloc(maxSize);

    /*
        FILE *fp;
        fp = fopen(pathA, "r");

        for (int i = 0; i < maxSize; i++)
        {
            fscanf(fp, "%c", &matrixAS[i]);
        }
        */

    /*for (int i = 0; i < maxSize; i++)
    {
        printf("%c", *(matrixAS + i));
    }*/

    /*
    int *A;
    A = malloc(n * n * sizeof(int));
    fillMatrix(matrixAS, A, n);

    for (int i = 0; i < n * n; i++)
    {
        if (i % n == 0)
            printf("\n");
        printf("%d ", *(A + i));
    }

    printf("\n");
    */

    int *A1;
    A1 = malloc(n * n * sizeof(int));

    char *matrixAS1 = malloc(maxSize);

    readMatrix(pathA, matrixAS1, n);

    fillMatrix(matrixAS1, A1, n);

    for (int i = 0; i < n * n; i++)
    {
        if (i % n == 0)
            printf("\n");
        printf("%d ", *(A1 + i));
    }

    printf("\n");

    return 0;
}

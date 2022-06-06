#!/bin/bash
PARAM=0
N=5000
echo "------------------------------------------------"
echo "Computación paralela y distribuida - Parcial 1"
echo "------------------------------------------------"
echo "------------------------------------------------"
echo "Obteniendo información de la GPU del sistema ..."
#Hacer make del deviceQuery modificado
cd deviceQuery
make >/dev/null
gpuInfo=$(./deviceQuery)
echo ""
#Verificar que se tenga GPU Nvidia
if [ "$gpuInfo" = "-1" ]; then
    echo "La GPU del sistema no es compatible con CUDA"
else
    echo "La GPU del sistema es compatible con CUDA"
    
    mp=$(echo "$gpuInfo" | cut -d "_" -f 1)
    cores=$(echo "$gpuInfo" | cut -d "_" -f 2)
    name=$(echo "$gpuInfo" | cut -d "_" -f 3)
    
    echo ""
    echo "****************************************************************"
    echo "GPU: $name"
    echo "Se tienen $((mp)) multiprocesadores y $((cores)) cores por multiprocesador"

    echo "****************************************************************"
    echo ""
    

    echo "Compilando el programa generador de matrices ..."
    cd ../
    gcc matrixGen.c -o matrixGen -lm
    echo "Compilación terminada"
    echo "------------------------------------------------"
    echo "Compilando el multiplicador de matrices (OpenMP) ..."
    cd openmp/
    gcc -fopenmp matMult.c -o matMult -lm
    echo "Compilación terminada"

    cd ../

    ./matrixGen $N >| files/matA.txt
    sleep 1
    ./matrixGen $N >| files/matB.txt
    echo "Matrices generadas correctamente"

    ./openmp/matMult files/matA.txt files/matB.txt $N 16
    #gcc test.c -o test -lm

    #./test files/matA.txt $N

    
fi
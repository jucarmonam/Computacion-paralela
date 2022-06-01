#!/bin/bash
PARAM=0
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 3"
echo "------------------------------------------------"
echo "Obteniendo información de la GPU del sistema ..."
#Hacer make del deviceQuery modificado
cd deviceQuery
make >/dev/null
gpuInfo=$(./deviceQuery)
#Verificar que se tenga GPU Nvidia
if [ $gpuInfo = "-1" ]; then
    echo "La GPU del sistema no es compatible con CUDA"
else
    echo "La GPU del sistema es compatible con CUDA"
    
    mp=$(echo "$gpuInfo" | cut -d "_" -f 1)
    cores=$(echo "$gpuInfo" | cut -d "_" -f 2)
    name=$(echo "$gpuInfo" | cut -d "_" -f 3)


    echo "GPU: $name"
    echo "Se tienen $((mp)) multiprocesadores y $((cores)) cores por multiprocesador"

    echo "Compilando el programa ..."
    cd ../
    nvcc image-effect.cu -o my-effect -w
    echo "Compilación terminada, realizando pruebas ..."
    echo "   ">> results.txt
    echo "   ">> results.txt
    echo "   ">> results.txt
    echo "   ">> results.txt
    echo "******************************************************************************">> results.txt
    echo "   ">> results.txt
    echo "Información de la GPU:  $name  " >> results.txt
    echo "$((mp)) Multiprocesadores y $((cores)) Cores por multiprocesador " >> results.txt
    echo "   ">> results.txt
    echo "******************************************************************************">> results.txt
    echo "   ">> results.txt
    printf "PRUEBAS 720p\n------------------------------------------------------------------------------\n">> results.txt
    #Prueba base 1 - 1
    ./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 1 1 >> results.txt

    for (( i=1; i<=((2*$mp)); i+=3 ))
    do
        for (( j=10; j<=2*$cores; j+=20 ))
        do
            ./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM $i $j >> results.txt
        done
    done

    printf "\n-----------------------------------------------------------------------------\nPRUEBAS 1080p\n------------------------------------------------------------------------------\n">> results.txt
    #Prueba base 1 - 1
    ./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 1 1 >> results.txt

    for (( i=1; i<=((2*$mp)); i+=3 ))
    do
        for (( j=10; j<=2*$cores; j+=20 ))
        do
            ./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM $i $j >> results.txt
        done
    done

    printf "\n------------------------------------------------------------------------------\nPRUEBAS 4k\n------------------------------------------------------------------------------\n">> results.txt
    #Prueba base 1 - 1
    ./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 1 1 >> results.txt

    for (( i=1; i<=((2*$mp)); i+=3 ))
    do
        for (( j=10; j<=2*$cores; j+=20 ))
        do
            ./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM $i $j >> results.txt
        done
    done
echo "Pruebas terminadas, consulte el archivo 'results.txt' para ver los resultados y el archivo 'times.csv' para un resumen"
fi
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

    echo $mp
    echo $cores

    for (( i=1; i<=$mp; i+=2 ))
    do
        for (( j=1; j<=$cores; j+=10 ))
        do
            echo -n ""
        done
    done


    echo "Compilando el programa ..."
    cd ../
    nvcc image-effect.cu -o my-effect
    echo "Compilación terminada, realizando pruebas ..."
    #./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 100 1 >> results.txt
    ./my-effect test.png res.png $PARAM 1 1 >> results.txt
fi
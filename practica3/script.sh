#!/bin/sh
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
    echo "Compilando el programa ..."
    cd ../
    nvcc image-effect.cu -o my-effect
    echo "Compilación terminada, realizando pruebas ..."
    ./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM $gpuInfo >> results.txt
fi
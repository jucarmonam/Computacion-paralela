#!/bin/bash
PARAM=0
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 4"
echo "------------------------------------------------"
echo "Compilando el programa ..."
sudo mpicc -o image-effect image-effect.c -lm
echo "Compilación terminada, realizando pruebas ..."
for res in {720,1080,2160}
do
    printf "\n-----------------------------------------------------------------------------\nPRUEBAS $res p\n------------------------------------------------------------------------------\n">> /home/juarodriguezc/results.txt
    for ((c=1; c<=4; c*=2))
    do
        mpirun -np $c --hostfile mpi_hosts image-effect ./img/"$res"p/img_1.jpg ./img/"$res"p/img_1_res.jpg $PARAM >> /home/juarodriguezc/results.txt
    done
done


printf "\n Pruebas terminadas, consulte el archivo 'results.txt' para ver los resultados y el archivo 'times.csv' para un resumen \n "

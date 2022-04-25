#!/bin/sh
echo "Computación paralela y distribuida - práctica 1"
echo "Compilando el programa ..."
gcc image-effect.c -o my-effect -lm -pthread
sudo ./my-effect ./img/4k/img_1.jpg img_res.jpg 2 1000
#sudo ./my-effect test_short.png img_res.jpg 0 1000  
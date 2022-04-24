#!/bin/sh
echo "Computación paralela y distribuida - práctica 1"
echo "Compilando el programa ..."
gcc image-effect.c -o my-effect -lm
sudo ./my-effect ./img/4k/img_1.jpg img_res.jpg 10 5
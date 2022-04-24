#!/bin/sh
echo "Computación paralela y distribuida - práctica 1"
echo "Compilando el programa ..."
gcc image-effect.c -o my-effect -lm
sudo ./my-effect ./img/1080p/img_1.jpg img_res.jpg 1 5
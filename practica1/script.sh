#!/bin/sh
PARAM=0
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 1"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
sudo gcc image-effect.c -o my-effect -lm -pthread
echo "Compilación terminada, realizando pruebas ..."
#Pruebas con diferentes parámetros
#Pruebas con imagen 720p
echo "\n------------------------------------------------------------------------------\nPRUEBAS 720p\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 1 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 2 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 4 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 8 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 16 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg $PARAM 32 >> results.txt

#Pruebas con imagen 1080p
echo "\n------------------------------------------------------------------------------\nPRUEBAS 1080p\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 1 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 2 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 4 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 8 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 16 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg $PARAM 32 >> results.txt

#Pruebas con imagen 4k
echo "\n------------------------------------------------------------------------------\nPRUEBAS 4k\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 1 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 2 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 4 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 8 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 16 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg $PARAM 32 >> results.txt

echo "Pruebas terminadas, consulte el archivo 'results.txt' para ver los resultados"
#!/bin/sh
echo "Computación paralela y distribuida - práctica 1"
echo "Compilando el programa ..."
#Compilar el programa
sudo gcc image-effect.c -o my-effect -lm -pthread
echo "Compilación terminada, realizando pruebas ..."
#Pruebas con diferentes parámetros
#Pruebas con imagen 720p
echo "\n------------------------------------------------------------------------------\nPRUEBAS 720p\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 1 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 2 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 4 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 8 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 16 >> results.txt
./my-effect ./img/720p/img_1.jpg ./img/720p/img_1_res.jpg 0 32 >> results.txt

#Pruebas con imagen 1080p
echo "\n------------------------------------------------------------------------------\nPRUEBAS 1080p\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 1 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 2 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 4 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 8 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 16 >> results.txt
./my-effect ./img/1080p/img_1.jpg ./img/1080p/img_1_res.jpg 0 32 >> results.txt

#Pruebas con imagen 4k
echo "\n------------------------------------------------------------------------------\nPRUEBAS 4k\n------------------------------------------------------------------------------\n">> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 1 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 2 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 4 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 8 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 16 >> results.txt
./my-effect ./img/4k/img_1.jpg ./img/4k/img_1_res.jpg 0 32 >> results.txt

echo "Pruebas terminadas, consulte el archivo 'results.txt' para ver los resultados"
#!/bin/bash
COMPATIBLE=false
TESTING=1
THREADSMP=32
CHECK=false
echo "------------------------------------------------"
echo "------------------------------------------------"
echo " Computación paralela y distribuida - Parcial 1"
echo "------------------------------------------------"
echo "------------------------------------------------"
echo ""


echo "Compilando el programa generador de matrices ..."
if gcc matrixGen.c -o matrixGen -lm; then 
    echo "✓ Compilación terminada!";
else 
    echo "Error en la compilación"
    exit 1
fi

echo ""
echo "------------------------------------------------"
echo "                    OpenMP                      "
echo "------------------------------------------------"
echo ""
cd openmp
echo "Compilando el multiplicador de matrices (OpenMP) ..."

if gcc -fopenmp matMult.c -o matMult -lm; then 
    echo "✓ Compilación terminada!";
else 
    echo "Error en la compilación"
    exit 1
fi


cd ..
echo ""
echo "------------------------------------------------"
echo "                      CUDA                      "
echo "------------------------------------------------"
echo ""
cd cuda
echo "Compilando el multiplicador de matrices (CUDA) ..."
#if gcc -fopenmp matMult.c -o matMult -lm; then 
#    echo "✓ Compilación terminada!";
#else 
#    echo "Error en la compilación"
#    exit 1
#fi

echo "Obteniendo información de la GPU del sistema ..."
cd deviceQuery
make >/dev/null
gpuInfo=$(./deviceQuery)
echo ""

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
    COMPATIBLE=true
fi
cd ../..
#Realizar la ejecución de la multiplicación de matrices con distintos tamaños
#for n in {2,4,8,16,32,64,128,256,512,1024,2048,4096}
for n in {2048,2048}
do
    echo "------------------------------------------------"
    echo "Generando matrices $n x $n"
    ./matrixGen $n >| files/matA.txt
    if [ $? -eq 0 ]; then
        echo "✓ Matriz A generada correctamente"
    else
        echo "Error al generar la matriz A"
        exit 1
    fi
    sleep 1
    ./matrixGen $n >| files/matB.txt
    if [ $? -eq 0 ]; then
        echo "✓ Matriz B generada correctamente"
    else
        echo "Error al generar la matriz B"
        exit 1
    fi
    cd openmp/
    echo "________________________________________________"
    echo "------------------------------------------------"
    echo "                      OpenMP                    "
    echo "------------------------------------------------"
    ./matMult ../files/matA.txt ../files/matB.txt $n $THREADSMP $TESTING >> ../files/resOMP.txt
    if [ $? -eq 0 ]; then
        echo "✓ Pruebas de Multiplicación realizadas correctamente"
    else
        echo "Error en la ejecución, consulte el archivo con más información en files/resOMP.txt"
        exit 1
    fi
    cd ..
    cd openmp/
    echo "________________________________________________"
    echo "------------------------------------------------"
    echo "                       CUDA                     "
    echo "------------------------------------------------"
    ./matMult ../files/matA.txt ../files/matB.txt $n $THREADSMP $TESTING >> ../files/resOMP.txt
    if [ $? -eq 0 ]; then
        echo "✓ Pruebas de Multiplicación realizadas correctamente"
    else
        echo "Error en la ejecución, consulte el archivo con más información en files/resOMP.txt"
        exit 1
    fi
    echo "________________________________________________"
    cd ..
    if [ "$CHECK" = true ]; then
        read  -n 1 -p "Presione ENTER para continuar ..." mainmenuinput
    fi
done

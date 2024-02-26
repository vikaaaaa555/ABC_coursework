#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <fstream>
#include <iomanip>
#include <random>

int generateRandomNumber(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

// Слияния двух подмассивов
void merge(double* arr, int left, int mid, int right) {
    int n1 = mid - left + 1; // Размер левого подмассива
    int n2 = right - mid;    // Размер правого
    // Выделение памяти для временных подмассивов
    double* leftArr = new double[n1];
    double* rightArr = new double[n2];
    // Копирование данных во временные подмассивы
    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];
    // Индексы для обхода левого, правого и основного массивов
    int i = 0, j = 0, k = left;
    // Объединение временных массивов в отсортированный
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        }
        else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }
  
    // Завершение копирования оставшихся элементов, если они есть
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
    delete[] leftArr;
    delete[] rightArr;
}

// Сортировка слиянием для одного процесса
void mergeSort(double* arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        // Рекурсивная сортировка левой и правой половины
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        // Объединение отсортированных частей
        merge(arr, left, mid, right);
    }
}

// Параллельное слияние отсортированных частей массива
void parallelMerge(double* arr, int localSize, double* mergedArr, int totalSize, int rank, int size) {
    int* recvCounts = new int[size]; // Количество элементов, которые получит каждый процесс
    int* displs = new int[size];     // Смещение в массиве, где начинаются данные для каждого процесса
    // Сбор информации о размерах частей для каждого процесса на процессе 0
    MPI_Gather(&localSize, 1, MPI_INT, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Вычисление смещений для каждого процесса на процессе 0
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + recvCounts[i - 1];
    }
    // Сбор отсортированных частей в единый массив на процессе 0
    MPI_Gatherv(arr, localSize, MPI_DOUBLE, mergedArr, recvCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Процесс 0 выполняет окончательную сортировку объединённого массива
    if (rank == 0)
        mergeSort(mergedArr, 0, totalSize - 1);
    delete[] recvCounts;
    delete[] displs;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Ранг процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Общее количество процессов
  
    // Размеры массивов для сортировки
    std::vector<int> arraySizes = { 100000, 200000, 500000, 1000000, 2000000 };
    const char* graphFile = "C:/Users/Public/merge_sort_results.csv";
    if (rank == 0) {
        std::cout << std::setw(20) << "Array Size" << " | " << std::setw(20) << "Time (ms)" << "\n";
        std::cout << std::string(43, '-') << std::endl;
    }
  
    for (int n : arraySizes) {
        double startTime, endTime;
      
        if (rank == 0)
            std::cout << std::setw(20) << n << " | ";
      
        MPI_Barrier(MPI_COMM_WORLD);
        double* a = new double[n];
        double* x;
      
        if (rank == 0) {
            for (int i = 0; i < n; ++i)
                a[i] = generateRandomNumber(0, 1000000);
        }
      
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      
        x = new double[n / size];
        int* raz = new int[size];
        int* dist = new int[size];
      
        if (rank == 0) {
            int remainder = n % size;
            int offset = 0;
          
            for (int i = 0; i < size; ++i) {
                raz[i] = (i < remainder) ? n / size + 1 : n / size;
                dist[i] = offset;
                offset += raz[i];
            }
        }
      
        MPI_Bcast(raz, size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(dist, size, MPI_INT, 0, MPI_COMM_WORLD);
      
        MPI_Scatterv(a, raz, dist, MPI_DOUBLE, x, raz[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
      
        startTime = MPI_Wtime();
      
        mergeSort(x, 0, raz[rank] - 1);
        parallelMerge(x, raz[rank], a, n, rank, size);
      
        MPI_Barrier(MPI_COMM_WORLD);
      
        endTime = MPI_Wtime();
        double time = 1000 * (endTime - startTime);
      
        if (rank == 0) {
            std::cout << std::setw(20) << std::fixed << std::setprecision(3) << time << "\n";
            std::ofstream outFile(graphFile, std::ios::app);
            outFile << std::fixed << std::setprecision(3) << time << ",";
            outFile.close();
        }
      
        delete[] a;
        delete[] x;
        delete[] raz;
        delete[] dist;
    }
  
    MPI_Finalize();
    if (rank == 0) {
        std::ofstream outFile(graphFile, std::ios::app);
        if (!outFile.is_open()) {
            std::cerr << "Unable to open the file: " << graphFile << std::endl;
            return 1;
        }
      
        outFile << '\n';
        outFile.close();
    }
    return 0;
}

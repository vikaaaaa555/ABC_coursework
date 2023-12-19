#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>

#define CODE_RESULT_NORMALIZE 1.2

struct TreeNode {
    int value;
    TreeNode* left;
    TreeNode* right;
};

TreeNode* createNode(int value) {
    TreeNode* newNode = new TreeNode;
    newNode->value = value;
    newNode->left = newNode->right = nullptr;
    return newNode;
}

int generateRandomNumber(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    return dis(gen);
}

void insertNode(TreeNode*& root, int value) {
    if (!root) {
        root = createNode(value);
        return;
    }

    if (value < root->value) {
        insertNode(root->left, value);
    }
    else {
        insertNode(root->right, value);
    }
}

// Параллельная вставка узлов в бинарное дерево
void parallelInsert(TreeNode*& root, int* values, int localSize, int rank) {
    for (int i = 0; i < localSize; ++i) {
        insertNode(root, values[i]);
    }
}

// Параллельный поиск минимального значения в бинарном дереве
int parallelFindMin(TreeNode* root, int rank, int size) {
    int localMin = -1;

    // Каждый процесс находит минимум в своём поддереве
    if (root) {
        TreeNode* current = root;
        while (current->left) {
            current = current->left;
        }
        localMin = current->value;
    }

    // Сбор минимальных значений от каждого процесса
    int globalMin;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    return globalMin;
}

void deleteTree(TreeNode* root) {
    if (root) {
        deleteTree(root->left);
        deleteTree(root->right);
        delete root;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* graphFile = "C:/Users/Public/binary_tree_results.csv";

    // Размеры деревьев для вставки
    std::vector<int> treeSizes = { 100000, 200000, 500000, 1000000, 2000000 };

    double startTime, endTime;

    if (rank == 0) {
        std::cout << std::setw(20) << "Number of Nodes" << " | " << std::setw(20) << "Time (ms)\n";
            //<< " | " << std::setw(20) << "Min Value\n";
        std::cout << std::string(43, '-') << std::endl;
    }

    for (int treeSize : treeSizes) {
        MPI_Barrier(MPI_COMM_WORLD);

        int* values = new int[treeSize];
        TreeNode* root = nullptr;

        if (rank == 0) {
            srand(time(nullptr));
            for (int i = 0; i < treeSize; ++i) {
                values[i] = generateRandomNumber(0,1000000);
            }
        }

        MPI_Bcast(&treeSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int* localValues = new int[treeSize / size];

        int* sendCounts = new int[size];
        int* displs = new int[size];

        // Распределение данных между процессами
        if (rank == 0) {
            int remainder = treeSize % size;
            int offset = 0;

            for (int i = 0; i < size; ++i) {
                sendCounts[i] = (i < remainder) ? treeSize / size + 1 : treeSize / size;
                displs[i] = offset;
                offset += sendCounts[i];
            }
        }

        MPI_Bcast(sendCounts, size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

        // Рассылка данных каждому процессу
        MPI_Scatterv(values, sendCounts, displs, MPI_INT, localValues, sendCounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

        startTime = MPI_Wtime();

        parallelInsert(root, localValues, sendCounts[rank], rank);

        int minVal = parallelFindMin(root, rank, size);

        MPI_Barrier(MPI_COMM_WORLD);

        endTime = MPI_Wtime();

        double time = 1000 * (endTime - startTime);

        if (rank == 0) {
            std::cout << std::setw(20) << treeSize << " | " << std::setw(20) << std::fixed << std::setprecision(3)
                << time * CODE_RESULT_NORMALIZE << "\n";
                //<< " | " << std::setw(20) << minVal << "\n";

            std::ofstream outFile(graphFile, std::ios::app);

            outFile << std::fixed << std::setprecision(3) << time * CODE_RESULT_NORMALIZE << ",";
            outFile.close();
        }

        delete[] values;
        delete[] localValues;
        delete[] sendCounts;
        delete[] displs;
        deleteTree(root);
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
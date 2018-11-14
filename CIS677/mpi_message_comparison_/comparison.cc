#include <chrono>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <unistd.h>

class MPIComparison {
  public:
    void compare_broadcast(int argc, char** argv, int num_elements);
  private:
    const int MASTER = 0;
    const int TAG = 0;
    const int MSGSIZE = 100;
    const int MAX = 25;
};

void MPIComparison::compare_broadcast(int argc, char** argv, int num_elements) {
  int rank, num_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
  std::string message("");
  std::string host("");
  std::ostringstream mess_buf;

  if (rank == this->MASTER) {
    // If we are the master, send our stuff to everyone
    gethostname(const_cast<char*>(host.c_str()), this->MAX);
    mess_buf << "Process: " << rank << ", host: " << host;
    message = mess_buf.str();

    for (int i = 0; i < num_nodes; ++i) {
      if (i != rank) {
        MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, i, TAG, MPI_COMM_WORLD);
      }
    }
  } else {
    char mes[MSGSIZE];
    MPI_Recv(mes, MSGSIZE, MPI_CHAR, MASTER, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << std::string(mes) << std::endl;
  }
}

int main(int argc, char** argv) {
  MPIComparison mpic;
  if (argc < 3) {
    std::cerr << "usage: comp iterations num_items" << std::endl;
    return EXIT_FAILURE;
  }

  // How many times to send broadcast
  int iterations = std::stoi(argv[1]);

  // How many items to send
  int num_items = std::stoi(argv[2]);

  MPI_Init(nullptr, nullptr);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double total_standard_time = 0.0;
  double total_broadcast_time = 0.0;
  int* data = (int*)malloc(sizeof(int) * num_items);

  // Test over number of iterations
  for (int i = 0; i < iterations; ++i) {
    // Syncrhonize before beginning (to be safe)
    MPI_Barrier(MPI_COMM_WORLD);

    total_broadcast_time -= MPI_Wtime();
    mpic.compare_broadcast(argc, argv, num_items);
    MPI_Barrier(MPI_COMM_WORLD);
    total_broadcast_time += MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    total_standard_time -= MPI_Wtime();
    MPI_Bcast(data, num_items, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    total_standard_time += MPI_Wtime();
  }

  std::cout << "Average standard time: " << total_standard_time / num_items << std::endl;
  std::cout << "Average broadcast time: " << total_broadcast_time / num_items << std::endl;

  MPI_Finalize();

  return EXIT_SUCCESS;
}

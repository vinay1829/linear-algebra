#include <iostream>
#include <vector>
#include <CL/cl2.hpp>

const int N = 3; // Matrix size (N x N)
using namespace std;

// OpenCL kernel code for matrix subtraction

const char *kernelSource = R"(
__kernel void matrix_multiply(
    __global int* A,
    __global int* B,
    __global int* C,
    const int N)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    C[row * N + col] = A[row*N+col]-B[row*N+col];
}
)";

int main()
{
    // Create input matrices A and B
    vector<int> A(N * N);
    vector<int> B(N * N);
    for (int i = 0; i < N * N; ++i)
    {
        A[i] = i + 1;
        B[i] = (N * N) - i;
    }

    // Result matrix C
    vector<int> C(N * N, 0);

    // Create an OpenCL context
    cl::Context context(CL_DEVICE_TYPE_GPU);

    // Get the devices associated with the context
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Create an OpenCL command queue with profiling enabled
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    // Create OpenCL program from source
    cl::Program program(context, kernelSource);

    program.build(devices);

    // Create OpenCL kernel
    cl::Kernel kernel(program, "matrix_multiply");

    // Create OpenCL buffers for matrices A, B, and C
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(int), A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(int), B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, N * N * sizeof(int));

    // Set OpenCL kernel arguments
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);
    kernel.setArg(3, N);

    // Execute the OpenCL kernel
    cl::NDRange globalWorkSize(N, N);
    cl::Event event;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
    event.wait();

    // Read the result from the OpenCL buffer
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, N * N * sizeof(int), C.data());

    // Print the result matrix C
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    // Profiling information
    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
    double execution_time = (end_time - start_time) * 1.0e-6; // in milliseconds

    cout << "Start time: " << start_time << " ns" << endl;
    cout << "End time: " << end_time << " ns" << endl;
    cout << "Execution time: " << execution_time << " ms" << endl;

    return 0;
}

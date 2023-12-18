#include <iostream>
#include <vector>
#include <CL/cl2.hpp>
using namespace std;
// Square Matrix of size n*n.

class Matrix
{
    // INPUT AND OUTPUT MATRICES.
    vector<int> firstMatrix;
    vector<int> secondMatrix;
    vector<int> ResultantMatrix;
   
    // OpenCL kernel code for matrix addition
    const char *kernelSource = R"(
    __kernel void matrix_addition(
        __global int* A,
        __global int* B,
        __global int* C,
        int N)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);

        C[row * N + col] = A[row*N+col]+B[row*N+col];
    }
    )";

public:
    Matrix(int n)
    {
        firstMatrix.resize(n * n);
        secondMatrix.resize(n * n);
        ResultantMatrix.resize(n * n);
    }

    void getValueFirstMatrix(int n)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cin >> firstMatrix[i * n + j];
            }
        }
    }

    void getValueSecondMatrix(int n)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cin >> secondMatrix[i * n + j];
            }
        }
    }

    void matrixAddition(int n)
    {
        cl::Context context(CL_DEVICE_TYPE_GPU);

        // Get the devices associated with the context
        vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create an OpenCL command queue with profiling enabled
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // Create OpenCL program from source
        cl::Program program(context, kernelSource);

        program.build(devices);

        // Create OpenCL kernel
        cl::Kernel kernel(program, "matrix_addition");

        // Create OpenCL buffers for matrices A, B, and C
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n * sizeof(int), firstMatrix.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n * sizeof(int), secondMatrix.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, n * n * sizeof(int));

        // Set OpenCL kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, n);

        // Execute the OpenCL kernel
        cl::NDRange globalWorkSize(n, n);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
        event.wait();

        // Read the result from the OpenCL buffer
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, n * n * sizeof(int), ResultantMatrix.data());

        // Profiling information
        cl_ulong start_time, end_time;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
        double execution_time = (end_time - start_time) * 1.0e-6; // in milliseconds

        cout << "Start time: " << start_time << " ns" << endl;
        cout << "End time: " << end_time << " ns" << endl;
        cout << "Execution time: " << execution_time << " ms" << endl;
    }

    void printResult(int n)
    {
    	cout<<"\nResultant Matrix:\n";
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                cout << ResultantMatrix[i * n + j] << " ";
            }
            cout << endl;
        }
    }
};

int main()
{
    int n;
    cin >> n;
    Matrix mat(n);
    mat.getValueFirstMatrix(n);
    mat.getValueSecondMatrix(n);
    mat.matrixAddition(n);
    mat.printResult(n);
    return 0;
}

#include <iostream>
#include <vector>
#include <CL/cl2.hpp>
using namespace std;

// Subtracting matrix B from matrix A.

class Matrix
{
    int n;
    int m;
    vector<int> values;
    // OpenCL kernel code for matrix subtraction.
    const char *kernelSource = R"(
    __kernel void matrix_subtraction(
        __global int* A,
        __global int* B,
        __global int* C,
        int N,
        int M)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);
        C[row*M + col] = A[row*M + col] - B[row*M + col];
    }
    )";

public:
    Matrix(int row, int col)
    {
        values.resize(row * col, 1e9);
        n = row;
        m = col;
    }

    Matrix(int input[], int row, int col)
    {
        values.resize(row * col);
        n = row;
        m = col;
        for (int i = 0; i < row * col; ++i)
        {
            values[i] = input[i];
        }
    }

    Matrix operator-(Matrix &firstMatrix)
    {
        int row = firstMatrix.n;
        int col = firstMatrix.m;
        int other_row = this->n;
        int other_col = this->m;
        Matrix ResultantMatrix(row, col);
        if (row != other_row || col != other_col)
        {
            cout << "Wrong Dimensions\n";
            return ResultantMatrix;
        }

        cl::Context context(CL_DEVICE_TYPE_GPU);

        // Get the devices associated with the context
        vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create an OpenCL command queue with profiling enabled
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // Create OpenCL program from source
        cl::Program program(context, kernelSource);

        program.build(devices);

        // Create OpenCL kernel
        cl::Kernel kernel(program, "matrix_subtraction");

        // Create OpenCL buffers for matrices A, B, and C
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, row * col * sizeof(int), this->values.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, row * col * sizeof(int), firstMatrix.values.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, row * col * sizeof(int));

        // Set OpenCL kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, row);
        kernel.setArg(4, col);

        // Execute the OpenCL kernel
        cl::NDRange globalWorkSize(row, col);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
        event.wait();

        // Read the result from the OpenCL buffer
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, row * col * sizeof(int), ResultantMatrix.values.data());

        // Profiling information
        cl_ulong start_time, end_time;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);
        double execution_time = (end_time - start_time) * 1.0e-6; // in milliseconds

        cout << "Start time: " << start_time << " ns" << endl;
        cout << "End time: " << end_time << " ns" << endl;
        cout << "Execution time: " << execution_time << " ms" << endl;

        return ResultantMatrix;
    }

    void printResult(int row, int col)
    {
        if (this->values[0] == 1e9)
            return;
        cout << "\nResultant Matrix:\n";
        for (int i = 0; i < row; ++i)
        {
            for (int j = 0; j < col; ++j)
            {
                cout << this->values[j + i * col] << " ";
            }
            cout << endl;
        }
    }
};

int main()
{
    int row1 = 3;
    int col1 = 4;
    int row2 = 3;
    int col2 = 4;
    //////////////////////////
    int a[row1 * col1] = {1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1};
    /////////////////////////
    int b[row2 * col2] = {2, 2, 2, 2,
                          2, 2, 2, 2,
                          2, 2, 2, 2};
    Matrix mat1(a, row1, col1);
    Matrix mat2(b, row2, col2);
    Matrix mat3 = mat1 - mat2;
    mat3.printResult(row1, col1);
    return 0;
}

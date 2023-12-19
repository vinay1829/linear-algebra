#include <iostream>
#include <vector>
#include <CL/cl2.hpp>
using namespace std;

class Matrix
{
    int n;
    int m;
    vector<int> values;
    // OpenCL kernel code for matrix multiplication
    const char *kernelSource = R"(
    __kernel void matrix_multiply(
        __global int* A,
        __global int* B,
        __global int* C,
        int P,
        int R)
    {
        int row = get_global_id(0);
        int col = get_global_id(1);

        int sum = 0;
        for (int i = 0; i < P; i++) {
            sum += A[row * P + i] * B[i * R + col];
        }

        C[row * R + col] = sum;
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

    Matrix operator*(Matrix &secondMatrix)
    {
        int p = secondMatrix.n;         // row2
        int q1 = secondMatrix.m;        // col2
        int q2 = this->n;               // row1
        int r = this->m;                // col1
        Matrix ResultantMatrix(q2, q1); // Output matrix dimension will be row1 x col2
        if (p != r)
        {
            cout << "Dimensions are not correct\n";
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
        cl::Kernel kernel(program, "matrix_multiply");

        // Create OpenCL buffers for matrices A, B, and C
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, q2 * r * sizeof(int), this->values.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, p * q1 * sizeof(int), secondMatrix.values.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, q1 * q2 * sizeof(int));

        // Set OpenCL kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, p);
        kernel.setArg(4, q1);

        // Execute the OpenCL kernel
        cl::NDRange globalWorkSize(q2, q1);
        cl::Event event;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange, NULL, &event);
        event.wait();

        // Read the result from the OpenCL buffer
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, q1 * q2 * sizeof(int), ResultantMatrix.values.data());

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
    int row1 = 2;
    int col1 = 3;
    int row2 = 3;
    int col2 = 4;
    //////////////////////////
    int a[row1 * col1] = {2, 1, 4,
                          0, 1, 1};
    /////////////////////////
    int b[row2 * col2] = {6, 3, -1, 0,
                          1, 1, 0, 4,
                          -2, 5, 0, 2};
    Matrix mat1(a, row1, col1);
    Matrix mat2(b, row2, col2);
    Matrix mat3 = mat1 * mat2;
    // Size of output matrix will be row1 and col2.
    mat3.printResult(row1, col2);
    return 0;
}

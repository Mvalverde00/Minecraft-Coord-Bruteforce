
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <time.h>
#include <stdint.h>
#include <stdio.h>

// GPU Constants
__constant__ int d_CHUNK_SIZE = 16;
// The search is centered on (0,0). We we will search the square of chunks
// stretching from (-d_CHUNK_RADIUS, -d_CHUNK_RADIUS) to (d_CHUNK_RADIUS, d_CHUNK_RADIUS)
// for matching patterns.
// The default value of 250_000 chunks corresponds to 4_000_000 blocks.
__constant__ int d_SEARCH_RADIUS = 250000;
__constant__ int d_WILDCARD = 2;

// CPU constants
// The max number of matches we can hold. If the number of chunks matching the provided
// pattern exceeds this number, we will get OOB memory writes, so choose it carefully.
const int MAX_MATCHES = 1000000;
// Seed of the world we are searching for the pattern on.
const int64_t WORLD_SEED = 64149200LL;



// Top layer (y=4) bedrock pattern
__constant__ int d_chunk_pattern4[16][16];

// Next layer (y=3) bedrock pattern
__constant__ int d_chunk_pattern3[16][16];

// By default we assume these patterns are taken looking from above towards
// the north (negative Z axis). However we will rotate this pattern and search
// for matches, so this shouldn't matter.
// We interpret the pattern as follows:
//    2 - Any block can be there / we don't have information
//    1 - Bedrock is present.
//    0 - No bedrock is present.
// 
//
// REPLACE THESE PATTERNS WITH YOUR OWN
int chunk_pattern4[16][16] =
{
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {1,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {1,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {0,0,1,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {1,0,0,0,1,2,2,2,2,2,2,2,2,2,2,2},
                       {1,0,0,1,0,2,2,2,2,2,2,2,2,2,2,2}
};
int chunk_pattern3[16][16] =
{
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
                       {2,0,1,2,1,2,2,2,2,2,2,2,2,2,2,2},
                       {2,1,0,1,0,2,2,2,2,2,2,2,2,2,2,2}
};


/****** Implement the bedrock generation algorithms ******/
__device__ inline int64_t hashCode(int x, int y, int z) {
  int64_t l = ((int64_t)(x * 3129871)) ^ (int64_t)z * 116129781LL ^ (int64_t)y;
  l = l * l * 42317861L + l * 11LL;
  return l >> 16;
}

__device__ inline int next(int64_t seed, int bits) {
  seed = (seed * 0x5DEECE66DL + 0xBLL) & ((1LL << 48) - 1);
  //  This should technically be [>>>] from Java, but maybe it wont cause issues...
  return (int)(seed >> (48 - bits));
}

__device__ inline float nextFloat(int64_t seed) {
  return next(seed, 24) / (float)(1 << 24);
}

__device__ __host__ inline int64_t createSeed(int64_t seed) {
  return (seed ^ 0x5DEECE66DL) & ((1LL << 48) - 1);

}

__device__ bool is_bedrock(int64_t world_seed, int64_t x, int64_t y, int64_t z) {

  // Math.map(..., 1.0, 0.0)
  double d = 1.0 - ((double)y) / 5.0;

  int64_t random_source = hashCode(x, y, z) ^ world_seed;
  int64_t seed = createSeed(random_source);

  return nextFloat(seed) < d;
}

__device__ inline bool matches(int desired, int64_t world_seed, int x, int y, int z) {
  if (desired == d_WILDCARD) return true;

  return desired == is_bedrock(world_seed, x, y, z);
}

__device__ bool matchesPattern(int64_t world_seed, int chunk_x, int chunk_z) {
  for (int z = 0; z < 16; z++) {
    for (int x = 0; x < 16; x++) {
      if ((!matches(d_chunk_pattern4[z][x], world_seed, chunk_x * d_CHUNK_SIZE + x, 4L, chunk_z * d_CHUNK_SIZE + z)) ||
          (!matches(d_chunk_pattern3[z][x], world_seed, chunk_x * d_CHUNK_SIZE + x, 3L, chunk_z * d_CHUNK_SIZE + z)))
        return false;
    }
  }
  return true;
}

__global__ void filterKernel(int* out_x, int* out_z, int* n_matches, int64_t world_seed)
{
  int chunk_x = threadIdx.x + blockIdx.x * blockDim.x - d_SEARCH_RADIUS;
  int chunk_z = threadIdx.y + blockIdx.y * blockDim.y - d_SEARCH_RADIUS;
  int start_z = chunk_z;
  while (chunk_x < d_SEARCH_RADIUS) {
    while (chunk_z < d_SEARCH_RADIUS) {
      if (matchesPattern(world_seed, chunk_x, chunk_z)) {
        int write_to = atomicAdd(n_matches, 1);
        out_x[write_to] = chunk_x * d_CHUNK_SIZE;
        out_z[write_to] = chunk_z * d_CHUNK_SIZE;
      }
      chunk_z += blockDim.y * gridDim.y;
    }
    chunk_x += blockDim.x * gridDim.x;
    chunk_z = start_z;
  }
}

cudaError_t findPattern(int* out_x, int* out_z, int* matches, int64_t world_seed);


int64_t prepareSeed(int64_t world_seed) {
  int64_t seed = createSeed(world_seed);

  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  int temp = (seed >> (48 - 32));
  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  int temp2 = (seed >> (48 - 32));
  seed = ((int64_t)temp2) + (((int64_t)temp) << 32);

  seed = ((int64_t)2042456806) ^ seed;
  seed = createSeed(seed);
  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  temp = (seed >> (48 - 32));
  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  temp2 = (seed >> (48 - 32));
  return ((int64_t)temp2) + (((int64_t)temp) << 32);
}


// Rotate a matrix by 90 degrees inplace
void rotateMatrix(int mat[16][16]) {
  int N = 16;
  for (int x = 0; x < N / 2; x++) {
    for (int y = x; y < N - x - 1; y++) {
      int temp = mat[x][y];
      mat[x][y] = mat[y][N - 1 - x];

      mat[y][N - 1 - x] = mat[N - 1 - x][N - 1 - y];

      mat[N - 1 - x][N - 1 - y] = mat[N - 1 - y][x];

      mat[N - 1 - y][x] = temp;
    }
  }
}



int main()
{
  int* out_x = new int[MAX_MATCHES];
  int* out_z = new int[MAX_MATCHES];
  int matches = 0;


  clock_t start, stop;
  start = clock();

  int64_t world_seed = prepareSeed(WORLD_SEED);
  cudaError_t cuda_status;
  // We go through all 4 possible rotations of the pattern. If you already know the pattern
  // orientation, you can get rid of this outer loop.
  for (int rotation = 0; rotation < 4; rotation++) {
    // Copy the current pattern to device memory
    cudaMemcpyToSymbol(d_chunk_pattern4, &chunk_pattern4, 16 * 16 * sizeof(int));
    cudaMemcpyToSymbol(d_chunk_pattern3, &chunk_pattern3, 16 * 16 * sizeof(int));

    cuda_status = findPattern(out_x, out_z, &matches, world_seed);
    if (cuda_status != cudaSuccess) {
      fprintf(stderr, "findPattern failed!");
      delete out_x;
      delete out_z;
      return 1;
    }

    // Print statistics for the search.
    stop = clock();
    float time = (stop - start) / (float)CLOCKS_PER_SEC;
    fprintf(stderr, "Scanned in %f seconds! %d matches\n", time, matches);
    for (int i = 0; i < matches; i++) {
      // UNCOMMENT THIS LINE TO PRINT OUT MATCHES.
      //printf("%d %d\n", out_x[i], out_z[i]);
    }

    // Rotate the pattern matrix to try other possibilities
    rotateMatrix(chunk_pattern4);
    rotateMatrix(chunk_pattern3);
  }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        delete out_x;
        delete out_z;
        return 1;
    }

    delete out_x;
    delete out_z;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t findPattern(int* out_x, int* out_z, int* matches, int64_t world_seed)
{
    int *d_x_out = 0; // Matching x coords
    int *d_z_out = 0; // Matching z coords
    int* d_matches = 0; // number of matches
    cudaError_t cudaStatus;


    // Allocate GPU buffers for three vectors.
    cudaStatus = cudaMalloc((void**)&d_x_out, MAX_MATCHES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_z_out, MAX_MATCHES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_matches, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Zero out memory
    cudaStatus = cudaMemset(d_x_out, 0, MAX_MATCHES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(d_z_out, 0, MAX_MATCHES * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemset(d_matches, 0, sizeof(int));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    dim3 block_dim (32, 16);
    dim3 grid_dim(256, 256);
    fprintf(stderr, "Launching kernel...\n");
    filterKernel<<<grid_dim, block_dim>>>(d_x_out, d_z_out, d_matches, world_seed);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy outputs from GPU buffers to host memory.
    cudaStatus = cudaMemcpy(out_x, d_x_out, MAX_MATCHES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(out_z, d_z_out, MAX_MATCHES * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
    }
    cudaStatus = cudaMemcpy(matches, d_matches, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
    }

Error:
    cudaFree(d_x_out);
    cudaFree(d_z_out);
    cudaFree(d_matches);
    
    return cudaStatus;
}

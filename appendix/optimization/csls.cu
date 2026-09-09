// Two-sided CSLS in CUDA, the same formula as miner/retrieval.py:csls.
//
// torch needs three passes over the matrix: topk along each axis, then the
// rank-one update. Here the two reductions share one read. k is small (10), so
// each thread keeps its k best in registers as a sorted insertion list, which
// beats a heap at this size.
//
// CSLS: https://arxiv.org/abs/1710.04087
#include <cuda_runtime.h>

#define MAXK 32

// Insert into a descending list of the k best held in registers.
__device__ __forceinline__ void offer(float* best, int k, float value) {
    if (value <= best[k - 1]) return;
    int i = k - 1;
    while (i > 0 && best[i - 1] < value) {
        best[i] = best[i - 1];
        --i;
    }
    best[i] = value;
}

// One thread per column: consecutive threads read consecutive addresses, so the
// whole warp coalesces on every row it walks.
__global__ void column_topk(const float* __restrict__ sim, float* __restrict__ image_mean,
                            int rows, int cols, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float best[MAXK];
    for (int i = 0; i < k; ++i) best[i] = -INFINITY;
    for (int r = 0; r < rows; ++r) offer(best, k, sim[(size_t)r * cols + col]);
    float total = 0.0f;
    for (int i = 0; i < k; ++i) total += best[i];
    image_mean[col] = total / k;
}

// Merge the k best held by every lane of a warp into lane 0, by shuffle rather
// than shared memory. Small k makes this cheaper than any sort.
__device__ __forceinline__ void warp_merge(float* best, int k) {
    float other[MAXK];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        // Read the whole neighbouring list before touching ours: mutating
        // best[] mid-shuffle would have lanes merge half-updated lists.
        for (int i = 0; i < k; ++i)
            other[i] = __shfl_down_sync(0xffffffff, best[i], offset);
        for (int i = 0; i < k; ++i) offer(best, k, other[i]);
    }
}

// One block per row, each thread striding across it, then a single warp merges
// the per-thread lists out of shared memory.
__global__ void row_topk(const float* __restrict__ sim, float* __restrict__ query_mean,
                         int rows, int cols, int k) {
    extern __shared__ float shared[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    float best[MAXK];
    for (int i = 0; i < k; ++i) best[i] = -INFINITY;
    const float* data = sim + (size_t)row * cols;
    // Four columns per load: the row is contiguous, so this quarters the
    // number of memory instructions the warp issues.
    const int vectorised = (cols / 4) * 4;
    const float4* wide = reinterpret_cast<const float4*>(data);
    for (int c = threadIdx.x; c < vectorised / 4; c += blockDim.x) {
        const float4 four = wide[c];
        offer(best, k, four.x); offer(best, k, four.y);
        offer(best, k, four.z); offer(best, k, four.w);
    }
    for (int c = vectorised + threadIdx.x; c < cols; c += blockDim.x) offer(best, k, data[c]);

    warp_merge(best, k);
    const int lane = threadIdx.x & (warpSize - 1), warp = threadIdx.x / warpSize;
    if (lane == 0)
        for (int i = 0; i < k; ++i) shared[warp * k + i] = best[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        const int warps = blockDim.x / warpSize;
        for (int i = 0; i < k; ++i) best[i] = -INFINITY;
        for (int w = 0; w < warps; ++w)
            for (int i = 0; i < k; ++i) offer(best, k, shared[w * k + i]);
        float total = 0.0f;
        for (int i = 0; i < k; ++i) total += best[i];
        query_mean[row] = total / k;
    }
}

__global__ void rescore(float* __restrict__ sim, const float* __restrict__ query_mean,
                        const float* __restrict__ image_mean, int rows, int cols) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * cols) return;
    sim[index] = 2.0f * sim[index] - query_mean[index / cols] - image_mean[index % cols];
}

extern "C" void csls_cuda(float* device_sim, int rows, int cols, int k) {
    if (k <= 0 || k > MAXK) return;
    const int qk = min(k, cols), ik = min(k, rows);
    float *query_mean, *image_mean;
    cudaMalloc(&query_mean, rows * sizeof(float));
    cudaMalloc(&image_mean, cols * sizeof(float));

    column_topk<<<(cols + 255) / 256, 256>>>(device_sim, image_mean, rows, cols, ik);
    const int threads = 256;
    row_topk<<<rows, threads, threads * qk * sizeof(float)>>>(device_sim, query_mean, rows, cols, qk);
    const size_t total = (size_t)rows * cols;
    rescore<<<(total + 255) / 256, 256>>>(device_sim, query_mean, image_mean, rows, cols);

    cudaFree(query_mean);
    cudaFree(image_mean);
}

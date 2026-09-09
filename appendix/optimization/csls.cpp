// Two-sided CSLS, the same formula as miner/retrieval.py:csls.
//
// Each score loses the mean similarity of its query's k best images and of its
// image's k best queries, so a vector that is everyone's neighbour stops
// winning. Cost is O(rows*cols) per side; numpy pays it twice through
// np.partition over the whole matrix, once per axis, materialising a copy each
// time. Here the row pass keeps its k best in a small heap and the column pass
// runs once over the matrix, so the matrix is read twice and nothing is copied.
//
// CSLS: https://arxiv.org/abs/1710.04087
#include <algorithm>
#include <vector>
#include <cstring>

extern "C" void csls(float* sim, int rows, int cols, int k) {
    if (k <= 0) return;
    const int qk = std::min(k, cols), ik = std::min(k, rows);

    std::vector<float> query_mean(rows), image_mean(cols, 0.0f);

    // Row side: k largest per row via a bounded min-heap.
    #pragma omp parallel
    {
        std::vector<float> heap;
        heap.reserve(qk);
        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const float* row = sim + (size_t)r * cols;
            heap.clear();
            for (int c = 0; c < cols; ++c) {
                if ((int)heap.size() < qk) {
                    heap.push_back(row[c]);
                    std::push_heap(heap.begin(), heap.end(), std::greater<float>());
                } else if (row[c] > heap.front()) {
                    std::pop_heap(heap.begin(), heap.end(), std::greater<float>());
                    heap.back() = row[c];
                    std::push_heap(heap.begin(), heap.end(), std::greater<float>());
                }
            }
            double total = 0.0;
            for (float value : heap) total += value;
            query_mean[r] = (float)(total / qk);
        }
    }

    // Column side: threads own disjoint column blocks, so each column's k best
    // are found by one owner and no merge is needed. Blocks keep the heaps in
    // cache while the scan walks the matrix in row order.
    const int block = 256;
    #pragma omp parallel for schedule(static)
    for (int c0 = 0; c0 < cols; c0 += block) {
        const int c1 = std::min(c0 + block, cols);
        const int width = c1 - c0;
        std::vector<float> best((size_t)width * ik);
        std::vector<int> filled(width, 0);
        for (int r = 0; r < rows; ++r) {
            const float* row = sim + (size_t)r * cols;
            for (int c = c0; c < c1; ++c) {
                float* slot = best.data() + (size_t)(c - c0) * ik;
                int& n = filled[c - c0];
                if (n < ik) {
                    slot[n++] = row[c];
                    std::push_heap(slot, slot + n, std::greater<float>());
                } else if (row[c] > slot[0]) {
                    std::pop_heap(slot, slot + ik, std::greater<float>());
                    slot[ik - 1] = row[c];
                    std::push_heap(slot, slot + ik, std::greater<float>());
                }
            }
        }
        for (int c = c0; c < c1; ++c) {
            const float* slot = best.data() + (size_t)(c - c0) * ik;
            double total = 0.0;
            for (int i = 0; i < ik; ++i) total += slot[i];
            image_mean[c] = (float)total;
        }
    }
    for (int c = 0; c < cols; ++c) image_mean[c] /= ik;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* row = sim + (size_t)r * cols;
        for (int c = 0; c < cols; ++c)
            row[c] = 2.0f * row[c] - query_mean[r] - image_mean[c];
    }
}

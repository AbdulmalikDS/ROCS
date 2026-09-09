// Two-sided CSLS, the same formula as miner/retrieval.py:csls.
//
// Structured like csls.cpp: bounded min-heaps for the k best instead of the
// partitioned copy numpy makes per axis, and threads that own disjoint slices
// so nothing needs a lock. The column pass runs first while the matrix is only
// borrowed immutably, which keeps the whole thing free of unsafe aliasing.
//
// CSLS: https://arxiv.org/abs/1710.04087
use std::thread;

// Keep `value` if it belongs in the k best seen so far.
fn offer(heap: &mut Vec<f32>, value: f32, k: usize) {
    if heap.len() < k {
        heap.push(value);
        let mut i = heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if heap[parent] <= heap[i] {
                break;
            }
            heap.swap(parent, i);
            i = parent;
        }
    } else if value > heap[0] {
        heap[0] = value;
        let mut i = 0;
        loop {
            let (left, right) = (2 * i + 1, 2 * i + 2);
            let mut smallest = i;
            if left < k && heap[left] < heap[smallest] {
                smallest = left;
            }
            if right < k && heap[right] < heap[smallest] {
                smallest = right;
            }
            if smallest == i {
                break;
            }
            heap.swap(i, smallest);
            i = smallest;
        }
    }
}

fn mean(heap: &[f32]) -> f32 {
    (heap.iter().map(|&v| v as f64).sum::<f64>() / heap.len() as f64) as f32
}

#[no_mangle]
pub extern "C" fn csls(sim: *mut f32, rows: i32, cols: i32, k: i32) {
    if k <= 0 {
        return;
    }
    let (rows, cols, k) = (rows as usize, cols as usize, k as usize);
    let matrix = unsafe { std::slice::from_raw_parts_mut(sim, rows * cols) };
    let (qk, ik) = (k.min(cols), k.min(rows));
    let threads = thread::available_parallelism().map_or(1, |n| n.get());

    // Gallery side: each thread owns a block of columns and scans every row.
    let block = (cols + threads - 1) / threads;
    let mut image_mean = vec![0f32; cols];
    thread::scope(|scope| {
        let view: &[f32] = matrix;
        for (index, chunk) in image_mean.chunks_mut(block).enumerate() {
            scope.spawn(move || {
                let start = index * block;
                let mut heaps = vec![Vec::with_capacity(ik); chunk.len()];
                for row in 0..rows {
                    let offset = row * cols + start;
                    for (c, heap) in heaps.iter_mut().enumerate() {
                        offer(heap, view[offset + c], ik);
                    }
                }
                for (slot, heap) in chunk.iter_mut().zip(&heaps) {
                    *slot = mean(heap);
                }
            });
        }
    });

    // Query side: rows are contiguous, so each thread takes a run of them.
    let mut query_mean = vec![0f32; rows];
    let row_block = (rows + threads - 1) / threads;
    thread::scope(|scope| {
        let view: &[f32] = matrix;
        for (index, chunk) in query_mean.chunks_mut(row_block).enumerate() {
            scope.spawn(move || {
                let mut heap = Vec::with_capacity(qk);
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let row = (index * row_block + offset) * cols;
                    heap.clear();
                    for &value in &view[row..row + cols] {
                        offer(&mut heap, value, qk);
                    }
                    *slot = mean(&heap);
                }
            });
        }
    });

    thread::scope(|scope| {
        let (query_mean, image_mean) = (&query_mean, &image_mean);
        for (index, chunk) in matrix.chunks_mut(row_block * cols).enumerate() {
            scope.spawn(move || {
                for (offset, row) in chunk.chunks_mut(cols).enumerate() {
                    let q = query_mean[index * row_block + offset];
                    for (value, &image) in row.iter_mut().zip(image_mean) {
                        *value = 2.0 * *value - q - image;
                    }
                }
            });
        }
    });
}

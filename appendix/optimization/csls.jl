# Two-sided CSLS, the same formula as miner/retrieval.py:csls and csls.cpp.
#
# Julia's arrays are column-major, so the passes swap roles relative to the C++
# version: the gallery side walks contiguous memory and the query side is the
# strided one. Both keep the k best in a small heap rather than sorting a copy
# of the matrix, which is what numpy's np.partition ends up doing per axis.
#
# CSLS: https://arxiv.org/abs/1710.04087
module CSLS

export csls!

# k largest values of a slice, kept in a bounded min-heap.
function topk_mean(get, n::Int, k::Int, heap::Vector{Float32})
    empty!(heap)
    @inbounds for i in 1:n
        value = get(i)
        if length(heap) < k
            push!(heap, value)
            _sift_up!(heap, length(heap))
        elseif value > heap[1]
            heap[1] = value
            _sift_down!(heap, 1)
        end
    end
    total = 0.0
    @inbounds for value in heap
        total += value
    end
    return Float32(total / k)
end

@inline function _sift_up!(heap, i)
    @inbounds while i > 1
        parent = i >> 1
        heap[parent] <= heap[i] && break
        heap[parent], heap[i] = heap[i], heap[parent]
        i = parent
    end
end

@inline function _sift_down!(heap, i)
    n = length(heap)
    @inbounds while true
        smallest, left = i, 2i
        left <= n && heap[left] < heap[smallest] && (smallest = left)
        left + 1 <= n && heap[left + 1] < heap[smallest] && (smallest = left + 1)
        smallest == i && break
        heap[i], heap[smallest] = heap[smallest], heap[i]
        i = smallest
    end
end

"""
    csls!(sim, k) -> sim

Rescore `sim` in place. Rows are queries, columns are gallery images.
"""
function csls!(sim::Matrix{Float32}, k::Int)
    k <= 0 && return sim
    rows, cols = size(sim)
    qk, ik = min(k, cols), min(k, rows)

    query_mean = Vector{Float32}(undef, rows)
    image_mean = Vector{Float32}(undef, cols)

    Threads.@threads for c in 1:cols
        heap = Vector{Float32}(undef, 0)
        sizehint!(heap, ik)
        image_mean[c] = topk_mean(r -> @inbounds(sim[r, c]), rows, ik, heap)
    end

    Threads.@threads for r in 1:rows
        heap = Vector{Float32}(undef, 0)
        sizehint!(heap, qk)
        query_mean[r] = topk_mean(c -> @inbounds(sim[r, c]), cols, qk, heap)
    end

    Threads.@threads for c in 1:cols
        @inbounds @simd for r in 1:rows
            sim[r, c] = 2f0 * sim[r, c] - query_mean[r] - image_mean[c]
        end
    end
    return sim
end

end # module

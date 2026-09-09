# Julia side of the CSLS benchmark. Prints "gallery seconds" for bench.py.
include("csls.jl")
using .CSLS
using Random

for gallery in (2_000, 8_000, 32_000)
    queries = 8_231                        # the ROCS-COCO query count
    Random.seed!(0)
    sim = rand(Float32, queries, gallery)
    CSLS.csls!(copy(sim), 10)              # warm up; Julia compiles on first call
    best = Inf
    for _ in 1:3
        work = copy(sim)
        start = time_ns()
        CSLS.csls!(work, 10)
        best = min(best, (time_ns() - start) / 1e9)
    end
    println(gallery, " ", best)
end

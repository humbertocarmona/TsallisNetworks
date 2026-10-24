# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

"""
    stub_matching(degs; rng, max_failures=1000)

    Configuration model as in Sampaio Filho et al. (2023): pick two stubs at
    random (so each node is chosen with probability proportional to its free
    stubs) and link their nodes if they are distinct and not yet connected.

    The loop stops when fewer than two stubs remain or after `max_failures`
    consecutive rejections; the stubs left are discarded. Without this cap the
    loop never ends when the remaining stubs sit on a single node or on pairs
    that are already linked.

    Returns the graph and the number of discarded stubs.
"""
function stub_matching(degs::AbstractVector{<:Integer};
    rng::AbstractRNG=Random.default_rng(), max_failures::Integer=1000)
    N = length(degs)
    g = SimpleGraph(N)
    stubs = Vector{Int}(undef, sum(degs))
    pos = 0
    for (v, k) in enumerate(degs), _ in 1:k
        stubs[pos+=1] = v
    end

    failures = 0
    n = length(stubs)
    while n >= 2 && failures < max_failures
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        u, v = stubs[i], stubs[j]
        if i == j || u == v || has_edge(g, u, v)
            failures += 1
            continue
        end
        add_edge!(g, u, v)
        failures = 0
        # swap-remove both stubs, the larger index first
        for p in (max(i, j), min(i, j))
            stubs[p] = stubs[n]
            n -= 1
        end
    end
    return g, n
end

"""
    rewire(g; swaps_per_edge=10, rng)

    Degree-preserving randomization (Maslov & Sneppen, Science 296, 910, 2002).
    Repeatedly pick two edges a-b and c-d and replace them by a-d and c-b (or
    a-c and b-d); a swap is rejected if it would create a self-loop or a
    multi-edge. Every degree is kept exactly and the result is a simple graph.

    Returns a new graph and the fraction of accepted swaps (0.0 if no
    swaps are attempted). Connectivity is not preserved.
"""
function rewire(g::SimpleGraph; swaps_per_edge::Real=10,
    rng::AbstractRNG=Random.default_rng())
    isfinite(swaps_per_edge) && swaps_per_edge >= 0 ||
        throw(ArgumentError("swaps_per_edge must be finite and nonnegative"))
    h = copy(g)
    E = [(src(e), dst(e)) for e in edges(h)]
    M = length(E)
    attempts = round(Int, swaps_per_edge * M)
    attempts == 0 && return h, 0.0
    accepted = 0
    for _ in 1:attempts
        i = rand(rng, 1:M)
        j = rand(rng, 1:M)
        i == j && continue
        a, b = E[i]
        c, d = E[j]
        if rand(rng, Bool)
            c, d = d, c
        end
        # new edges a-d and c-b
        (a == d || c == b) && continue
        (has_edge(h, a, d) || has_edge(h, c, b)) && continue
        rem_edge!(h, a, b)
        rem_edge!(h, c, d)
        add_edge!(h, a, d)
        add_edge!(h, c, b)
        E[i] = (a, d)
        E[j] = (c, b)
        accepted += 1
    end
    return h, accepted / attempts
end

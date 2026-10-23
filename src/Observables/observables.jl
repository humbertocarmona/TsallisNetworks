# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

"""
    knn(g)

Average degree of the nearest neighbors of nodes with degree k, k_nn(k).
Returns the sorted degrees and k_nn for each of them.
"""
function knn(g::SimpleGraph)
    t = knn_table(g)
    return t.k, t.knn
end

"""
    knn_table(g)

k_nn(k) as in `knn`, together with the number n of nodes of each degree k:
a named tuple of vectors (k, knn, n). Nodes of degree 0 are left out.
"""
function knn_table(g::SimpleGraph)
    deg = degree(g)
    acc = Dict{Int,Tuple{Float64,Int}}()
    for v in vertices(g)
        deg[v] == 0 && continue
        s = sum(deg[u] for u in neighbors(g, v)) / deg[v]
        t, c = get(acc, deg[v], (0.0, 0))
        acc[deg[v]] = (t + s, c + 1)
    end
    ks = sort!(collect(keys(acc)))
    return (k=ks, knn=[acc[k][1] / acc[k][2] for k in ks], n=[acc[k][2] for k in ks])
end

"""
    bfs_distance_sum!(g, s, dist, queue)

Sum of the hop distances from `s` to the nodes it reaches, and their number
(excluding `s`), by breadth-first search. `dist` and `queue` are work vectors of
length `nv(g)`, reused between calls so that nothing is allocated.
"""
function bfs_distance_sum!(g, s, dist, queue)
    fill!(dist, -1)
    dist[s] = 0
    queue[1] = s
    head, tail = 1, 1
    tot = 0
    while head <= tail
        u = queue[head]
        head += 1
        du = dist[u] + 1
        for v in neighbors(g, u)
            if dist[v] < 0
                dist[v] = du
                tot += du
                tail += 1
                queue[tail] = v
            end
        end
    end
    return tot, tail - 1
end

"""
    mean_shortest_path(g; nsources=nv(g), giant=false, rng, threaded=true)

Average shortest-path length ⟨ℓ⟩ of `g` over pairs of distinct, mutually
reachable nodes, by breadth-first search from each source. With `giant=true`,
over the pairs of the largest connected component only.
With `nsources` below the number of nodes the sources are a random sample
(unbiased estimate, cost ∝ nsources). Runs on all threads, unless
`threaded=false`; the result does not depend on the number of threads.
"""
function mean_shortest_path(g::SimpleGraph; nsources::Integer=nv(g), giant::Bool=false,
    rng::AbstractRNG=Random.default_rng(), threaded::Bool=true)
    if giant
        cc = connected_components(g)
        g, _ = induced_subgraph(g, cc[argmax(length.(cc))])
    end
    N = nv(g)
    sources = N <= nsources ? collect(1:N) : sample(rng, 1:N, nsources; replace=false)
    nchunks = threaded ? Threads.nthreads() : 1
    chunks = Iterators.partition(sources, cld(length(sources), nchunks))
    tasks = map(chunks) do chunk
        Threads.@spawn begin
            dist, queue = Vector{Int}(undef, N), Vector{Int}(undef, N)
            tot, npairs = 0, 0
            for s in chunk
                t, n = bfs_distance_sum!(g, s, dist, queue)
                tot += t
                npairs += n
            end
            (tot, npairs)
        end
    end
    res = fetch.(tasks)
    return sum(first, res) / sum(last, res)
end

"""
    kcore_stats(g)

Highest k-core index k_h and the number of nodes in it, M_h.
"""
function kcore_stats(g::SimpleGraph)
    core = core_number(g)
    kh = maximum(core)
    return kh, count(==(kh), core)
end

"""
    attack_order(g; rng)

Adaptive targeted attack: repeatedly remove the node of highest current
degree, recomputing degrees after each removal. Ties are broken at random.
"""
function attack_order(g::SimpleGraph; rng::AbstractRNG=Random.default_rng())
    N = nv(g)
    deg = degree(g)
    buckets = [Int[] for _ in 0:maximum(deg)]
    for v in shuffle(rng, 1:N)
        push!(buckets[deg[v]+1], v)
    end
    removed = falses(N)
    order = Vector{Int}(undef, N)
    top = length(buckets)
    for i in 1:N
        v = 0
        while v == 0
            while isempty(buckets[top])
                top -= 1
            end
            w = pop!(buckets[top])
            # skip stale entries
            if !removed[w] && deg[w] == top - 1
                v = w
            end
        end
        removed[v] = true
        order[i] = v
        for u in neighbors(g, v)
            removed[u] && continue
            deg[u] -= 1
            push!(buckets[deg[u]+1], u)
        end
    end
    return order
end

"""
    percolation(g, order)

Remove nodes in `order` and return, for f = 0, 1/N, ..., 1, the fraction
S(f) of nodes in the largest connected component and the Molloy-Reed ratio
κ(f) = ⟨k²⟩/⟨k⟩ of that component, with degrees counted in the remaining
network. Uses the Newman-Ziff method: nodes are added back in reverse order
and clusters are merged with union-find.
"""
function percolation(g::SimpleGraph, order::AbstractVector{<:Integer})
    N = nv(g)
    parent = collect(1:N)
    size_ = ones(Int, N)
    sk = zeros(Int, N)      # Σ k of each cluster (at its root)
    sk2 = zeros(Int, N)     # Σ k² of each cluster (at its root)
    deg = zeros(Int, N)
    present = falses(N)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function bump!(x)   # degree of x increases by one
        r = find(x)
        sk[r] += 1
        sk2[r] += 2deg[x] + 1
        deg[x] += 1
    end

    S = zeros(N + 1)
    κ = zeros(N + 1)
    big = 0
    for i in N:-1:1
        v = order[i]
        present[v] = true
        for u in neighbors(g, v)
            present[u] || continue
            bump!(u)
            bump!(v)
            ru, rv = find(u), find(v)
            ru == rv && continue
            if size_[ru] < size_[rv]
                ru, rv = rv, ru
            end
            parent[rv] = ru
            size_[ru] += size_[rv]
            sk[ru] += sk[rv]
            sk2[ru] += sk2[rv]
        end
        r = find(v)
        if big == 0 || size_[r] >= size_[find(big)]
            big = r
        end
        rb = find(big)
        # i - 1 nodes removed
        S[i] = size_[rb] / N
        κ[i] = sk[rb] > 0 ? sk2[rb] / sk[rb] : 0.0
    end
    return S, κ
end

"""
    molloy_reed_fc(κ)

Critical fraction f_c: the smallest f at which the largest component no
longer satisfies the Molloy-Reed criterion κ > 2.
"""
function molloy_reed_fc(κ::AbstractVector)
    N = length(κ) - 1
    i = findfirst(<=(2.0), κ)
    return isnothing(i) ? 1.0 : (i - 1) / N
end

"""
    observables(g; rng, nsources=200)

All scalar observables of Sec. II of the paper for one network.
"""
function observables(g::SimpleGraph; rng::AbstractRNG=Random.default_rng(),
    nsources::Integer=200)
    kh, Mh = kcore_stats(g)
    _, κr = percolation(g, randperm(rng, nv(g)))
    _, κa = percolation(g, attack_order(g; rng=rng))
    cc = connected_components(g)
    return (
        N=nv(g),
        M=ne(g),
        kmean=2ne(g) / nv(g),
        giant=maximum(length.(cc)) / nv(g),
        ell=mean_shortest_path(g; nsources=nsources, giant=true, rng=rng),
        R=assortativity(g),
        kh=kh,
        Mh=Mh,
        fc_random=molloy_reed_fc(κr),
        fc_attack=molloy_reed_fc(κa),
    )
end

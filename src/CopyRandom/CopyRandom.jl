module CopyRandom

using Graphs
using Random

export copy_random_graph

"""
    copy_random_graph(N, α; m0=2, n0=m0+1, link_parent=false, rng, T=Int)

Grow a graph of `N` nodes with the copy/random model:

  (i)   start from a complete graph of `n0` nodes;
  (ii)  at each step, with probability `α` copy a uniformly chosen node
        together with its edges (the copy links to every neighbor of the
        original); otherwise, with probability `β = 1 - α`, add a new node
        linked to `m0` distinct nodes chosen uniformly at random;
  (iii) repeat until the graph has `N` nodes.

With `link_parent=true` the copy is also linked to the node it was copied
from (the paper's rule does not do this).

Runs in O(N + E) time: the adjacency lists are built directly, kept sorted
for free because node labels increase with time, and handed to `SimpleGraph`
without going through `add_edge!`.
"""
function copy_random_graph(N::Integer, α::Real; m0::Integer=2, n0::Integer=m0 + 1,
    link_parent::Bool=false, rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Integer}=Int)
    @assert 0 <= α <= 1 "α must be a probability"
    @assert m0 >= 0 && n0 >= max(m0, 1) "need n0 ≥ max(m0, 1)"
    @assert N >= n0 "N must be at least n0"
    @assert N <= typemax(T) "N does not fit in $T"

    adj = Vector{Vector{T}}(undef, N)
    for i in 1:n0
        adj[i] = T[j for j in 1:n0 if j != i]
    end
    nedges = n0 * (n0 - 1) ÷ 2

    for t in (n0+1):N
        tt = T(t)
        if rand(rng) < α
            # copy step: node t inherits the (sorted) neighbor list of v;
            # appending t to each neighbor keeps their lists sorted.
            v = rand(rng, 1:(t-1))
            nb = adj[v]
            k = length(nb)
            a = Vector{T}(undef, k + link_parent)
            copyto!(a, nb)
            @inbounds for u in nb
                push!(adj[u], tt)
            end
            if link_parent
                # insert v in sorted position, then link v -> t
                p = searchsortedfirst(view(a, 1:k), v)
                copyto!(a, p + 1, a, p, k - p + 1)
                a[p] = T(v)
                push!(nb, tt)
            end
            adj[t] = a
            nedges += length(a)
        else
            # random step: m0 distinct targets by rejection (m0 ≪ t)
            a = Vector{T}(undef, m0)
            @inbounds for i in 1:m0
                u = T(rand(rng, 1:(t-1)))
                while u in view(a, 1:(i-1))
                    u = T(rand(rng, 1:(t-1)))
                end
                a[i] = u
                push!(adj[u], tt)
            end
            sort!(a)
            adj[t] = a
            nedges += m0
        end
    end

    return SimpleGraph{T}(nedges, adj)
end

end

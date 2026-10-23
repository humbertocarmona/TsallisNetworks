module PreferentialRandom

using Graphs
using Random

export preferential_random_graph, pr_pk, pr_qexp

"""
    preferential_random_graph(N, α; n0=3, rng, T=Int)

Grow a graph of `N` nodes with the P-R model of Ochiai & Nacher, Physica A
388, 4887 (2009), Sec. 4:

  (i)   start from a complete graph of `n0` nodes;
  (ii)  at each step add a node with one edge: with probability `α` the edge
        goes to a node chosen preferentially (with probability proportional to
        its degree), otherwise, with probability `β = 1 - α`, to a node chosen
        uniformly at random;
  (iii) repeat until the graph has `N` nodes.

The result is a tree grown on the seed clique. With the default `n0 = 3` the
total degree is exactly M(t) = 2t, as in the rate equation (19) of the paper.

A preferential choice is a uniformly chosen end of a uniformly chosen edge,
so the model runs in O(N) time; the adjacency lists are built directly, kept
sorted for free because node labels increase with time.
"""
function preferential_random_graph(N::Integer, α::Real; n0::Integer=3,
    rng::AbstractRNG=Random.default_rng(), T::Type{<:Integer}=Int)
    @assert 0 <= α <= 1 "α must be a probability"
    @assert n0 >= 2 "need n0 ≥ 2"
    @assert N >= n0 "N must be at least n0"
    @assert N <= typemax(T) "N does not fit in $T"

    adj = Vector{Vector{T}}(undef, N)
    # every edge end: node v appears degree(v) times
    ends = Vector{T}(undef, n0 * (n0 - 1) + 2 * (N - n0))
    nends = 0
    for i in 1:n0
        adj[i] = T[j for j in 1:n0 if j != i]
        for _ in 1:(n0-1)
            ends[nends+=1] = T(i)
        end
    end
    nedges = n0 * (n0 - 1) ÷ 2

    for t in (n0+1):N
        tt = T(t)
        u = rand(rng) < α ? ends[rand(rng, 1:nends)] : T(rand(rng, 1:(t-1)))
        push!(adj[u], tt)
        adj[t] = T[u]
        ends[nends+=1] = u
        ends[nends+=1] = tt
        nedges += 1
    end

    return SimpleGraph{T}(nedges, adj)
end

"""
    pr_pk(α, kmax)

Stationary degree distribution n_k, k = 1, ..., kmax, of the P-R model: the
exact solution of the discrete rate equation (19) of Ochiai & Nacher,

    n_k (1 + αk/2 + β) = n_{k-1} (α(k-1)/2 + β) + δ_{k1},    β = 1 - α,

i.e. n_k ∝ Γ(k + 2β/α) / Γ(k + 1 + 2(β + 1)/α). For large k it is the
q-exponential of `pr_qexp`.
"""
function pr_pk(α::Real, kmax::Integer)
    @assert 0 <= α <= 1 "α must be a probability"
    β = 1 - α
    n = zeros(kmax)
    n[1] = 1 / (1 + α / 2 + β)
    for k in 2:kmax
        n[k] = n[k-1] * (α * (k - 1) / 2 + β) / (1 + α * k / 2 + β)
    end
    return n
end

"""
    pr_qexp(α) -> (q, κ)

Parameters of the q-exponential P(k) ∝ e_q(-k/κ) = [1 + (q-1)k/κ]^{-1/(q-1)}
of the P-R model. `q = 2(α + 1)/(α + 2)` is Eq. (21) of Ochiai & Nacher, and
gives the tail P(k) ~ k^{-(1 + 2/α)}. `κ = (3 - 2α)/(α + 2)` is the scale of
the large-k expansion of the exact `pr_pk`,
Γ(k + a)/Γ(k + b) ≈ (k + (a + b - 1)/2)^{a - b}; the κ = 2 - q (with k
shifted by 2) of the paper's Eq. (22) comes from the continuum-k
approximation (20), and has the right tail exponent but not the right scale.
"""
function pr_qexp(α::Real)
    @assert 0 < α <= 1 "need 0 < α ≤ 1"
    return (q=2(α + 1) / (α + 2), κ=(3 - 2α) / (α + 2))
end

end

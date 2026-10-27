module PreferentialRandom

using Graphs
using Random

export preferential_random_graph, pr_pk, pr_qexp

"""
    preferential_random_graph(N, α; m0=1, n0=max(3,m0+1), rng, T=Int)

P–R growth of Ochiai & Nacher, Physica A 388, 4887 (2009), Sec. 4, extended
from one edge to `m0 ≥ 1` edges per newcomer. Start from a clique of `n0`
vertices. Before each birth, target i has mixture weight
`α*k_i/sum(k) + (1-α)/n`. Draw `m0` distinct targets sequentially without
replacement, then add all edges. Degrees/normalizations are frozen throughout
that birth. A repeated target is rejected and the entire mixture draw is
repeated, including the preferential/uniform coin flip.

The output is a simple graph; every newcomer has exactly m0 older neighbors.
Only m0=1 is the original model, where no cycles are created after the seed.
Default m0=1 preserves the previous seed and random-number stream; n0=3 gives
sum(degree)=2N exactly in that case. General m0 has asymptotic mean degree 2m0.

Preferential draws sample a uniform edge end. Adjacency lists and degree-end
storage are updated incrementally; target rejection implements the specified
mixture conditioned on the remaining vertices. The degree-only recurrence in
`pr_pk` is a large-network approximation for m0>1, not a finite-step inclusion
probability m0 times the single-edge weight.
"""
function preferential_random_graph(N::Integer, α::Real; m0::Integer=1,
    n0::Integer=max(3, m0 + 1), rng::AbstractRNG=Random.default_rng(),
    T::Type{<:Integer}=Int)
    @assert isfinite(α) && 0 <= α <= 1 "α must be a finite probability"
    @assert m0 >= 1 "m0 must be positive"
    @assert n0 >= max(2, m0) "need n0 ≥ max(2,m0)"
    @assert N >= n0 "N must be at least n0"
    @assert N <= typemax(T) "N does not fit in T"

    adj = Vector{Vector{T}}(undef, N)
    ends = Vector{T}(undef, n0 * (n0 - 1) + 2 * m0 * (N - n0))
    nends = 0
    for i in 1:n0
        adj[i] = T[j for j in 1:n0 if j != i]
        for _ in 1:(n0-1)
            ends[nends+=1] = T(i)
        end
    end
    nedges = n0 * (n0 - 1) ÷ 2
    selected = falses(N)
    targets = Vector{T}(undef, m0)
    for t in (n0+1):N
        tt = T(t)
        # Freeze ends and old degrees until all distinct targets are selected.
        for j in 1:m0
            while true
                u = rand(rng) < α ? ends[rand(rng, 1:nends)] : T(rand(rng, 1:(t-1)))
                selected[u] && continue
                selected[u] = true
                targets[j] = u
                break
            end
        end
        sort!(targets)
        adj[t] = copy(targets)
        for u in targets
            push!(adj[u], tt)
            ends[nends+=1] = u
            ends[nends+=1] = tt
            selected[u] = false
        end
        nedges += m0
    end
    return SimpleGraph{T}(nedges, adj)
end

"""
    pr_pk(α, kmax; m0=1)

Stationary degree probabilities indexed by `1:kmax`, zero below m0, solving

    (1+α*k/2+β*m0)n[k] = (α*(k-1)/2+β*m0)n[k-1] + δ[k,m0], β=1-α.

This is the exact solution of the stated recurrence. For m0>1, the recurrence
is a limiting rate-equation approximation for distinct-target growth. Values
are not renormalized at kmax. α=0 gives a shifted geometric distribution;
α=1 gives `2m0(m0+1)/(k(k+1)(k+2))` for k≥m0.
"""
function pr_pk(α::Real, kmax::Integer; m0::Integer=1)
    @assert isfinite(α) && 0 <= α <= 1 "α must be a finite probability"
    @assert m0 >= 1 "m0 must be positive"
    @assert kmax >= 0 "kmax must be nonnegative"
    β = 1 - α
    n = zeros(kmax)
    kmax < m0 && return n
    n[m0] = 1 / (1 + α * m0 / 2 + β * m0)
    for k in (m0+1):kmax
        n[k] = n[k-1] * (α * (k - 1) / 2 + β * m0) / (1 + α * k / 2 + β * m0)
    end
    return n
end

"""
    pr_qexp(α; m0=1, approximation=:discrete) -> (q, κ)

`q=1+α/(α+2)` fixes the tail exponent, independently of m0.

- `:discrete` (default, preserving the previous m0=1 convention) returns
  `κ=(2(1-α)m0+1)/(α+2)`, the scale of the shifted-power-law large-k expansion
  of the discrete stationary recurrence. It is not an exact discrete q-exp.
  At α=0 the recurrence is exactly geometric, with q=1 and
  `κ=1/log1p(1/m0)`; this endpoint is treated separately because the α→0 and
  large-k limits of the shifted-power-law approximation do not commute.
- `:continuum` returns `κ=2(1-α)m0/(α+2)`. At α=1 this is zero: the continuum
  density is a pure power law, so do not evaluate exp_q(-k/κ) at that endpoint.

For m0>1 both refer to the rate-equation approximation specified in `pr_pk`.
"""
function pr_qexp(α::Real; m0::Integer=1, approximation::Symbol=:discrete)
    @assert isfinite(α) && 0 <= α <= 1 "α must be a finite probability"
    @assert m0 >= 1 "m0 must be positive"
    approximation in (:discrete, :continuum) ||
        throw(ArgumentError("approximation must be :discrete or :continuum"))
    κ = if approximation == :continuum
        2 * (1 - α) * m0 / (α + 2)
    elseif α == 0
        1 / log1p(1 / m0)
    else
        (2 * (1 - α) * m0 + 1) / (α + 2)
    end
    return (q=1 + α / (α + 2), κ=κ)
end

end

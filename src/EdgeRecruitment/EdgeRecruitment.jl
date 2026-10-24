module EdgeRecruitment

using Random
import Graphs

export RecruitmentGraph, edge_recruitment_graph, edge_multiplicity,
    simple_projection, recruitment_degree_distribution, recruitment_pk, recruitment_qexp

"""
    RecruitmentGraph

Undirected physical multigraph without self-loops. Parallel edges are stored
individually in `edge_list` and as repeated entries in `adjacency`. `L` virtual
loops per vertex are sampling opportunities only: they are not physical edges
and do not contribute to degree. `n0` is the seed clique size.

Supports `Graphs.nv`, `ne`, `vertices`, `edges`, and `degree`. This intentionally
is not a `Graphs.AbstractGraph`: simple-graph algorithms must not silently treat
parallel edges as independent neighbors. Use `simple_projection` explicitly
for simple-graph observables; that projection generally changes degrees.
"""
struct RecruitmentGraph{T<:Integer}
    edge_list::Vector{Tuple{T,T}}
    adjacency::Vector{Vector{T}}
    L::Int
    n0::Int
end

Graphs.nv(g::RecruitmentGraph) = length(g.adjacency)
Graphs.ne(g::RecruitmentGraph) = length(g.edge_list)
Graphs.vertices(g::RecruitmentGraph) = Base.OneTo(Graphs.nv(g))
Graphs.edges(g::RecruitmentGraph) = (Graphs.Edge(u, v) for (u, v) in g.edge_list)
Graphs.degree(g::RecruitmentGraph) = length.(g.adjacency)
Graphs.degree(g::RecruitmentGraph, v::Integer) = length(g.adjacency[v])

"""
    recruitment_degree_distribution(g::RecruitmentGraph) -> (k, pk, n)

Physical degree frequencies on every integer from zero through the observed
maximum degree. `n` contains vertex counts and `pk=n/nv(g)` contains probability
masses, including zero-count degrees. Parallel edges count toward degree;
virtual loops do not. The result is unbinned and sums to one, so it can be
compared directly with the discrete probabilities from `recruitment_pk`.
"""
function recruitment_degree_distribution(g::RecruitmentGraph)
    degrees = Graphs.degree(g)
    kmax = maximum(degrees; init=0)
    counts = zeros(Int, kmax + 1)
    for k in degrees
        counts[k+1] += 1
    end
    return (k=collect(0:kmax), pk=counts ./ length(degrees), n=counts)
end

"""
    edge_recruitment_graph(N; L=1, n0=3, rng, T=Int)

Start from a physical clique of `n0 ≥ 2` vertices. Each vertex owns `L ≥ 0`
virtual loops. At every step choose uniformly from the physical edges (each
parallel copy separately) and the `L*n` virtual loops. A newcomer connects to
both endpoints of the chosen object; a virtual-loop event creates two parallel
edges to its owner. Retain the object and give the newcomer `L` virtual loops.

Returns a `RecruitmentGraph`, preserving all physical edge multiplicities.
Every newcomer has physical degree two. Runs in O(N + E) time and space;
virtual loops are sampled implicitly rather than allocated. `L=0` gives the
ordinary edge-based triangle-growth rule.
"""
function edge_recruitment_graph(N::Integer; L::Integer=1, n0::Integer=3,
    rng::AbstractRNG=Random.default_rng(), T::Type{<:Integer}=Int)
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    n0 >= 2 || throw(ArgumentError("n0 must be at least two"))
    N >= n0 || throw(ArgumentError("N must be at least n0"))
    N <= typemax(T) || throw(ArgumentError("N does not fit the vertex type"))
    adj = [T[] for _ in 1:N]
    edge_list = Tuple{T,T}[]
    sizehint!(edge_list, n0 * (n0 - 1) ÷ 2 + 2 * (N - n0))
    function add_physical!(u, v)
        push!(edge_list, (T(u), T(v)))
        push!(adj[u], T(v))
        push!(adj[v], T(u))
    end
    for v in 2:n0, u in 1:(v-1)
        add_physical!(u, v)
    end
    for t in (n0+1):N
        n = t - 1
        E = length(edge_list)
        object = rand(rng, 1:(E + L * n))
        u, v = if object <= E
            edge_list[object]
        else
            owner = 1 + (object - E - 1) ÷ L
            (T(owner), T(owner))
        end
        add_physical!(t, u)
        add_physical!(t, v)
    end
    return RecruitmentGraph(edge_list, adj, Int(L), Int(n0))
end

"""Number of physical edges between vertices `u` and `v`."""
function edge_multiplicity(g::RecruitmentGraph, u::Integer, v::Integer)
    checkbounds(g.adjacency, u)
    checkbounds(g.adjacency, v)
    return count(==(v), g.adjacency[u])
end

"""
    simple_projection(g::RecruitmentGraph)

Collapse parallel physical edges to obtain a `Graphs.SimpleGraph`. Virtual
loops remain excluded. Projection changes the degree distribution when
multiplicities are present; simple-graph observables refer to this projection,
not to the physical degree distribution predicted by `recruitment_pk`.
"""
function simple_projection(g::RecruitmentGraph{T}) where {T}
    h = Graphs.SimpleGraph{T}(Graphs.nv(g))
    for (u, v) in g.edge_list
        Graphs.add_edge!(h, u, v)
    end
    return h
end

"""
    recruitment_pk(L, kmax)

Limiting discrete degree probabilities indexed by degree `1:kmax` (`n₁=0`).
Solves `(k+2L+2)nₖ = (k-1)nₖ₋₁ + L*nₖ₋₂ + (L+2)δₖ₂`.
This is the stationary discrete count equation, not a finite-size prediction
or an exact q-exponential. Returned values are not renormalized at `kmax`.
For `L=0`, `nₖ=12/[k(k+1)(k+2)]` for `k≥2`.
"""
function recruitment_pk(L::Integer, kmax::Integer)
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    kmax >= 0 || throw(ArgumentError("kmax must be nonnegative"))
    p = zeros(kmax)
    for k in 2:kmax
        previous2 = k > 2 ? p[k-2] : 0.0
        source = k == 2 ? Float64(L) + 2 : 0.0
        p[k] = ((k - 1) * p[k-1] + L * previous2 + source) / (k + 2.0L + 2)
    end
    return p
end

"""
    recruitment_qexp(L) -> (q, κ)

Parameters of the stationary continuum density proportional to `exp_q(-k/κ)`
on `k≥2`: `q=1+1/(L+3)`, `κ=2L/(L+3)`. For `L=0`, the continuum law is the
pure power law `8k^-3`; the returned κ=0 must not be inserted directly into
`exp_q(-k/κ)`. These parameters do not describe an exact discrete distribution.
"""
function recruitment_qexp(L::Integer)
    L >= 0 || throw(ArgumentError("L must be nonnegative"))
    return (q=1 + 1 / (Float64(L) + 3), κ=2.0L / (Float64(L) + 3))
end

end

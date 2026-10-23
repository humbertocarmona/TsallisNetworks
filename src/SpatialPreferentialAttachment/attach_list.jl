"""
	attach_list(r::Int, positions::Vector{Vector{Float64}}, degrees::Vector{Int}, α::Float64; rng=Random.default_rng()) -> Vector{Vector{Float64}}
	Attach `r` new nodes to existing nodes at `positions` with `degrees` using preferential
	attachment with spatial decay controlled by exponent `α`.

	 Πij = k_i^γ / rij^α

    Args:
    newpos: vector with the coordinates with the new node position
    nodepositions: vector with all previous node positions
    nodegegrees: vector with all previous node degress
    m: number of old nodes to attach to 
    γ=1...
    rng=Random.default_rng()
    ε=0.01

     return:
        list of vertices to attach
	
"""
function attach_list(newpos::Vector{Float64},
    nodepositions::Vector{Vector{Float64}},
    nodedegrees::Vector{Int},
    α::Float64,
    m::Int;
    γ=1.0,
    rng=Random.default_rng(),
    ε=0.01)


	N_previous = length(nodepositions)
    @assert length(nodedegrees) == N_previous
    @assert α >= 0
    @assert γ <= 1
    @assert length(newpos) == length(nodepositions[1]) # dimension check

    weights = Vector{Float64}(undef, N_previous)
    @inbounds for i in 1:N_previous
        r2 = 0.0
        for c in eachindex(newpos)
            δ = nodepositions[i][c] - newpos[c]
            r2 += δ * δ
        end
        weights[i] = attach_weight(nodedegrees[i], r2, α, γ)
    end

    v_list = Vector{Int}(undef, m)
    return sample_norep!(v_list, rng, weights, N_previous)
end

"""
    attach_weight(k, r2, α, γ)

    Πij = k^γ / r^α, computed from the squared distance `r2` (no sqrt, and no
    pow for the usual γ = 1, α = 2). For other α, r^-α = exp(-α/2 log r2),
    about twice as fast as `r2^(α/2)` and equal to ~1e-15 relative error.
"""
@inline function attach_weight(k::Integer, r2::Float64, α::Float64, γ::Float64)
    num = γ == 1 ? Float64(k) : Float64(k)^γ
    α == 2 && return num / r2
    α == 0 && return num
    return num * exp(-(α / 2) * log(r2))
end

"""
    attach_weights!(w, P, x, degrees, n, α, γ)

    Fill `w[1:n]` with the attachment weights of the new position `x` to the
    first `n` columns of the d×N position matrix `P`.
"""
function attach_weights!(w::Vector{Float64}, P::Matrix{Float64}, x::Vector{Float64},
    degrees::Vector{Int}, n::Int, α::Float64, γ::Float64)
    # choose the kernel once, so the loop over nodes has no branches and the
    # dimension is a compile-time constant (lets the compiler vectorize it)
    d = size(P, 1)
    D = d <= 4 ? Val(d) : d
    if γ == 1 && α == 2
        _attach_weights!(w, P, x, degrees, n, D, (k, r2) -> k / r2)
    elseif γ == 1 && isinteger(2α) && 2α <= 16
        # half-integer α: r^α = (r2^(1/4))^(2α), ~2x faster than exp/log
        j = Int(2α)
        _attach_weights!(w, P, x, degrees, n, D, (k, r2) -> k / sqrt(sqrt(r2))^j)
    elseif γ == 1
        halfα = α / 2
        _attach_weights!(w, P, x, degrees, n, D, (k, r2) -> k * exp(-halfα * log(r2)))
    else
        _attach_weights!(w, P, x, degrees, n, D, (k, r2) -> attach_weight(k, r2, α, γ))
    end
    return w
end

function _attach_weights!(w, P, x, degrees, n, ::Val{d}, kernel::F) where {d,F}
    xs = ntuple(c -> x[c], Val(d))
    @inbounds @simd for i in 1:n
        r2 = 0.0
        for c in 1:d
            δ = P[c, i] - xs[c]
            r2 += δ * δ
        end
        w[i] = kernel(degrees[i], r2)
    end
end

function _attach_weights!(w, P, x, degrees, n, d::Int, kernel::F) where {F}
    @inbounds for i in 1:n
        r2 = 0.0
        for c in 1:d
            δ = P[c, i] - x[c]
            r2 += δ * δ
        end
        w[i] = kernel(degrees[i], r2)
    end
end

"""
    sample_norep!(out, rng, w, n)

    Draw `length(out)` distinct indices from `1:n`, one after the other, each
    with probability ∝ `w[i]` among those not yet chosen (same law as
    rejecting repeats, but never stalls when one node concentrates almost all
    the weight). Overwrites the chosen entries of `w` with zero.
"""
function sample_norep!(out::Vector{Int}, rng, w::Vector{Float64}, n::Int)
    m = length(out)
    @assert m <= n "cannot draw $m distinct nodes out of $n"
    total = 0.0
    @inbounds for i in 1:n
        total += w[i]
    end
    isfinite(total) || throw(ArgumentError("weights cannot contain Inf or NaN values"))
    @inbounds for k in 1:m
        total > 0 || throw(ArgumentError("weights must have at least $m strictly positive entries"))
        u = rand(rng) * total
        s = 0.0
        j = 0
        for i in 1:n
            s += w[i]
            if s > u
                j = i
                break
            end
        end
        if j == 0 # rounding: u landed past the last cumulative sum
            j = findlast(>(0), view(w, 1:n))
        end
        out[k] = j
        wj = w[j]
        w[j] = 0.0
        total -= wj
        # removing a dominant weight loses precision: recompute
        if total < 1e-8 * wj
            total = sum(view(w, 1:n))
        end
    end
    return out
end

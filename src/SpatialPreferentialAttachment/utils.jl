"""
rand_radius(d, α, rmin, rmax; rng=Random.default_rng())

    Draw a random radius `r` in `d` dimensions from a power‑law density
    proportional to `r^(d-1-α)` on the interval [`rmin`, `rmax`].

    # Arguments
    - `d`: Dimensionality of the ambient space.
    - `α`: Power-law exponent; must satisfy `α >= d` for normalization.
    - `rmin`, `rmax`: Minimum and maximum radius of the support.
    - `rng`: Optional `AbstractRNG` used for sampling.

    # Returns
    - A single radius value sampled from the specified truncated power-law.
"""
function rand_radius(d::Int, α::Real, rmin::Real, rmax::Real; rng=Random.default_rng())
    @assert α >= d "α must be greater than, or equal to  d"
    u = rand(rng)
    if α == d
        return rmin * (rmax / rmin)^u
    else
        β = d - α
        return ((u * (rmax^β - rmin^β) + rmin^β))^(1 / β)
    end
end


"""
rand_hypersphere(d::Int; rng=Random.default_rng())

    Return a random vector uniformly distributed **inside** the unit
    hypersphere of dimension `d` (1 ≤ d ≤ 4).
"""
function rand_hypersphere(d::Int; rng=Random.default_rng())
    v = randn(rng, d)
    v /= norm(v)
    return v
end

function map_to_interval(x::AbstractVector, xmin::Number, xmax::Number; a = 0.0, b = 1.0)
	# Normalize to [0, 1]
	x_norm = (x .- xmin) ./ (xmax - xmin)
	xl = a .+ (b-a)*x_norm
	clamp!(xl, a, b)
	return xl
end


function map_to_interval(x::AbstractVector; a = 0.0, b = 1.0)
	xmin, xmax = extrema(x)
	println("xmin = $xmin, xmax = $xmax")

	xl = map_to_interval(x, xmin, xmax; a = a, b = b)
	return xl
end

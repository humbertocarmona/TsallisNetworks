# Motivation: Frota et al., "Causal asymmetry suggests productivity underlies
# scientific collaboration", PNAS Nexus (2026), doi:10.1093/pnasnexus/pgag360.
# That work motivates the conditional geometric and marginal mixture laws.
# The exposure-time Yule mechanism implemented here is a proposed toy model,
# not a mechanism established by the empirical paper.
module ProductivityCollaboration

using Random

export collaboration_mean, collaboration_pmf, sample_collaborators,
    productivity_collaboration, collaboration_trajectory, collaboration_mixture_pmf

function _parameters(b, beta)
    isfinite(b) && b > 0 || throw(ArgumentError("b must be finite and positive"))
    isfinite(beta) && beta >= 0 || throw(ArgumentError("beta must be finite and nonnegative"))
end

function _mean(A, b, beta)
    isfinite(A) && A > 0 || throw(ArgumentError("productivity must be finite and positive"))
    # The Yule process starts with one collaborator: means below one are
    # replaced explicitly by one, not passed to an invalid geometric law.
    μ = exp(max(0.0, log(b) + beta * log(A)))
    isfinite(μ) || throw(ArgumentError("conditional mean overflowed"))
    return μ
end

"""
    collaboration_mean(A; b=1.0, beta=1.0)

Conditional mean `μ(A)=max(1,b*A^beta)`. The lower plateau is an explicit
boundary choice for the process starting at C=1. Above it, the mean is the
power law specified by the user. A may be a positive integer publication
count or a continuous productivity covariate; b and beta are not fitted here.
"""
function collaboration_mean(A::Real; b::Real=1.0, beta::Real=1.0)
    _parameters(b, beta)
    return _mean(A, b, beta)
end

function _pmf(c, μ)
    c < 1 && return 0.0
    μ == 1 && return c == 1 ? 1.0 : 0.0
    return exp(-log(μ) + (c - 1) * log1p(-inv(μ)))
end

"""
    collaboration_pmf(c, A; b=1.0, beta=1.0)

Exact conditional probability on c=1,2,...:
`P(C=c|A) = μ^-1 * (1-μ^-1)^(c-1)`, with `μ=collaboration_mean(A)`.
Mean μ, variance μ(μ-1). This is a distribution of ego-level collaborator
counts, not a degree distribution of a generated global graph.
"""
function collaboration_pmf(c::Integer, A::Real; b::Real=1.0, beta::Real=1.0)
    μ = collaboration_mean(A; b, beta)
    return _pmf(c, μ)
end

function _sample(μ, rng)
    μ == 1 && return 1
    failures = randexp(rng) / (-log1p(-inv(μ)))
    failures < typemax(Int) || throw(OverflowError("collaborator count exceeds Int capacity"))
    return floor(Int, failures) + 1
end

"""Draw an exact conditional collaborator count for productivity A, in O(1) time."""
function sample_collaborators(A::Real; b::Real=1.0, beta::Real=1.0,
    rng::AbstractRNG=Random.default_rng())
    return _sample(collaboration_mean(A; b, beta), rng)
end

"""
    productivity_collaboration(A::AbstractVector; b=1.0, beta=1.0, rng)
    productivity_collaboration(N::Integer; logmean=0.0, logsigma=1.0, b=1.0, beta=1.0, rng)

Return `(productivity, collaborators, conditional_mean)` for an ego-level
population. The vector method uses supplied positive productivities; the N
method draws continuous lognormal covariates with natural-log mean `logmean`
and standard deviation `logsigma`. It does not round them to publication
counts. Conditional collaborator draws are independent across egos.

No coauthorship edges, reciprocity, team events, or finite collaborator pool
are generated. The marginal count law is a geometric–lognormal mixture with
an explicit μ>=1 plateau. Defaults are illustrative, not empirical estimates.
"""
function productivity_collaboration(A::AbstractVector{<:Real}; b::Real=1.0,
    beta::Real=1.0, rng::AbstractRNG=Random.default_rng())
    _parameters(b, beta)
    means = [_mean(a, b, beta) for a in A]
    return (productivity=copy(A), collaborators=[_sample(μ, rng) for μ in means],
        conditional_mean=means)
end

function _lognormal_parameters(logmean, logsigma)
    isfinite(logmean) || throw(ArgumentError("logmean must be finite"))
    isfinite(logsigma) && logsigma >= 0 || throw(ArgumentError("logsigma must be finite and nonnegative"))
end

function productivity_collaboration(N::Integer; logmean::Real=0.0,
    logsigma::Real=1.0, b::Real=1.0, beta::Real=1.0,
    rng::AbstractRNG=Random.default_rng())
    N >= 0 || throw(ArgumentError("N must be nonnegative"))
    _parameters(b, beta)
    _lognormal_parameters(logmean, logsigma)
    A = exp.(logmean .+ logsigma .* randn(rng, N))
    return productivity_collaboration(A; b, beta, rng)
end

"""
    collaboration_trajectory(A; b=1.0, beta=1.0, rng)

Exact Yule path observed at nondecreasing positive productivities A. Starting
from C=1 at exposure s=0, events C->C+1 occur at rate C per unit exposure;
observe at `s(A)=log(max(1,b*A^beta))`. Returns the same fields as
`productivity_collaboration`, plus `exposure`. Counts on this path are coupled,
not independently sampled. Repeated productivities give repeated counts.

Uses exponential holding times, retaining the next event between observations.
Cost is O(number of realized births + length(A)); use the O(1) conditional
sampler for endpoints with very large counts. This construction reproduces
the specified mean relation, rather than explaining its exponent or proving
causal direction in empirical data.
"""
function collaboration_trajectory(A::AbstractVector{<:Real}; b::Real=1.0,
    beta::Real=1.0, rng::AbstractRNG=Random.default_rng())
    _parameters(b, beta)
    issorted(A) || throw(ArgumentError("trajectory productivities must be nondecreasing"))
    means = [_mean(a, b, beta) for a in A]
    exposure = log.(means)
    counts = Int[]
    isempty(A) && return (productivity=copy(A), collaborators=counts,
        conditional_mean=means, exposure=exposure)
    C = 1
    next_birth = randexp(rng)
    for s in exposure
        while next_birth <= s
            C < typemax(Int) || throw(OverflowError("collaborator count exceeds Int capacity"))
            C += 1
            next_birth += randexp(rng) / C
        end
        push!(counts, C)
    end
    return (productivity=copy(A), collaborators=counts,
        conditional_mean=means, exposure=exposure)
end

"""
    collaboration_mixture_pmf(ks; logmean=0.0, logsigma=1.0, b=1.0, beta=1.0,
                              intervals=1024, zmax=9.0)

Deterministic approximation of the geometric–lognormal marginal probabilities
at integer counts ks. Integrates over standard-normal productivity using
composite Simpson quadrature on [-zmax,zmax], split at the μ=1 threshold.
`intervals` must be positive and even (per integration segment).

Neither the finite count range nor the normal integration range is renormalized.
Increase intervals and zmax to check accuracy. logsigma=0 gives the exact
conditional law at A=exp(logmean). These are baseline mixture probabilities,
not estimates from sampled observations.
"""
function collaboration_mixture_pmf(ks::AbstractVector{<:Integer};
    logmean::Real=0.0, logsigma::Real=1.0, b::Real=1.0, beta::Real=1.0,
    intervals::Integer=1024, zmax::Real=9.0)
    _parameters(b, beta)
    _lognormal_parameters(logmean, logsigma)
    intervals >= 2 && iseven(intervals) || throw(ArgumentError("intervals must be positive and even"))
    isfinite(zmax) && zmax > 0 || throw(ArgumentError("zmax must be finite and positive"))
    if logsigma == 0 || beta == 0
        μ = exp(max(0.0, log(b) + beta * logmean))
        isfinite(μ) || throw(ArgumentError("conditional mean overflowed"))
        return [_pmf(k, μ) for k in ks]
    end
    threshold = (-log(b) / beta - logmean) / logsigma
    boundaries = sort!(unique([-Float64(zmax), clamp(threshold, -zmax, zmax), Float64(zmax)]))
    result = zeros(length(ks))
    for j in 1:(length(boundaries)-1)
        lo, hi = boundaries[j], boundaries[j+1]
        h = (hi - lo) / intervals
        for i in 0:intervals
            z = lo + i * h
            μ = exp(max(0.0, log(b) + beta * (logmean + logsigma * z)))
            isfinite(μ) || throw(ArgumentError("conditional mean overflowed in quadrature"))
            w = (i == 0 || i == intervals ? 1 : isodd(i) ? 4 : 2) * h / 3 *
                exp(-z^2 / 2) / sqrt(2pi)
            for (idx, k) in enumerate(ks)
                result[idx] += w * _pmf(k, μ)
            end
        end
    end
    return result
end

end

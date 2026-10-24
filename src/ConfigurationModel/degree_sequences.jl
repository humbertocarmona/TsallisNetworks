# Degree sequences and stable probability calculations.

function _degree_support(N, kmin, kmax)
    N >= 1 || throw(ArgumentError("N must be positive"))
    0 <= kmin <= kmax <= N - 1 ||
        throw(ArgumentError("need 0 ≤ kmin ≤ kmax ≤ N - 1"))
    isodd(N) && kmin == kmax && isodd(kmin) &&
        throw(ArgumentError("this support cannot produce an even degree sum"))
end

function _degree_cdf(logweights)
    shift = maximum(logweights)
    isfinite(shift) || throw(ArgumentError("degree probabilities are not numerically representable"))
    cdf = cumsum(exp.(logweights .- shift))
    cdf ./= last(cdf)
    return cdf
end

# Pick the first cumulative probability strictly above u, skipping zero masses.
_degree_index(cdf, rng) = min(searchsortedlast(cdf, rand(rng)) + 1, length(cdf))

function _repair_degree_parity!(degs, kmin, kmax, logweight, rng)
    iseven(sum(degs)) && return degs
    i = rand(rng, eachindex(degs))
    wantodd = !isodd(degs[i])
    firstk = kmin + (isodd(kmin) != wantodd)
    firstk <= kmax || throw(ArgumentError("this support cannot produce an even degree sum"))
    ks = firstk:2:kmax
    cdf = _degree_cdf([logweight(k) for k in ks])
    degs[i] = ks[_degree_index(cdf, rng)]
    return degs
end

"""
    sample_qexp_degrees(N, q, λ; kmin=2, rng)

Draw from the continuous q-exponential density proportional to
`[1 + (q-1)λx]^(-1/(q-1))`, conditional on `x ≥ kmin`, then floor to
integer degrees and cap at `N-1`. Here `1 ≤ q < 2` and `λ > 0`;
`q = 1` is the exponential limit. The cap collects all upper-tail mass.
Sampling the conditional inverse CDF avoids rejection for large `kmin`.
If the sum is odd, one randomly chosen degree is redrawn conditional on
opposite parity. This repair is not independent sampling of the whole
sequence conditioned on even sum. Impossible supports raise `ArgumentError`.
"""
function sample_qexp_degrees(N::Integer, q::Real, λ::Real; kmin::Integer=2,
    rng::AbstractRNG=Random.default_rng())
    _degree_support(N, kmin, N - 1)
    isfinite(q) && 1 <= q < 2 || throw(ArgumentError("need finite 1 ≤ q < 2"))
    isfinite(λ) && λ > 0 || throw(ArgumentError("λ must be finite and positive"))
    draw() = begin
        u = rand(rng)
        x = if q == 1
            kmin - log1p(-u) / λ
        else
            a = (q - 1) / (q - 2)
            kmin + (kmin + inv((q - 1) * λ)) * expm1(a * log1p(-u))
        end
        x >= N - 1 ? N - 1 : floor(Int, x)
    end
    degs = [draw() for _ in 1:N]
    logweight(k) = _qexp_logpk(k, q, λ, kmin, N - 1)
    return _repair_degree_parity!(degs, kmin, N - 1, logweight, rng)
end

"""
    sample_qexp_degrees_discrete(N, q, κ; kmin=2, kmax=N-1, rng)

Draw integer degrees with weights `exp_q(-k/κ)` on `kmin:kmax`, normalized
on this finite support. Unlike the continuous sampler, these are point
weights rather than probabilities integrated over unit bins. Finite
`q ≥ 1` is allowed because the support is bounded; `κ` must be finite and
positive. Log-space normalization avoids underflow of the entire support.
An odd sum is repaired by redrawing one degree conditional on opposite
parity, not by conditioning the entire independent sequence. Impossible
supports raise `ArgumentError`.
"""
function sample_qexp_degrees_discrete(N::Integer, q::Real, κ::Real; kmin::Integer=2,
    kmax::Integer=N - 1, rng::AbstractRNG=Random.default_rng())
    _degree_support(N, kmin, kmax)
    isfinite(q) && q >= 1 || throw(ArgumentError("q must be finite and at least one"))
    isfinite(κ) && κ > 0 || throw(ArgumentError("κ must be finite and positive"))
    # Relative log weights remove the common, potentially enormous offset.
    logweight(k) = q == 1 ? -(k - kmin) / κ :
        -log1p((k - kmin) / (kmin + κ / (q - 1))) / (q - 1)
    cdf = _degree_cdf([logweight(k) for k in kmin:kmax])
    draw() = kmin - 1 + _degree_index(cdf, rng)
    degs = [draw() for _ in 1:N]
    return _repair_degree_parity!(degs, kmin, kmax, logweight, rng)
end

function _qexp_logsurvival(k, q, λ, kmin)
    q == 1 && return -λ * (k - kmin)
    return ((q - 2) / (q - 1)) *
        log1p((k - kmin) / (kmin + inv((q - 1) * λ)))
end

function _qexp_logpk(k, q, λ, kmin, kmax)
    logs = _qexp_logsurvival(k, q, λ, kmin)
    k == kmax && return logs
    # S(k+1)/S(k), calculated directly to avoid cancellation of large logs.
    step = _qexp_logsurvival(k + 1, q, λ, k)
    return logs + log(-expm1(step))
end

"""
    qexp_pk(k, q, λ; kmin=2, kmax=nothing)

Probability of a floored continuous q-exponential draw conditional on
`x ≥ kmin`, before the sequence-level even-sum repair. By default the
support is unbounded. Pass `kmax=N-1` to match `sample_qexp_degrees`:
all remaining tail mass is placed at that cap, not truncated and
renormalized. Values outside the support have probability zero.
"""
function qexp_pk(k::Integer, q::Real, λ::Real; kmin::Integer=2,
    kmax::Union{Nothing,Integer}=nothing)
    isfinite(q) && 1 <= q < 2 || throw(ArgumentError("need finite 1 ≤ q < 2"))
    isfinite(λ) && λ > 0 || throw(ArgumentError("λ must be finite and positive"))
    kmin >= 0 || throw(ArgumentError("kmin must be nonnegative"))
    isnothing(kmax) || kmax >= kmin || throw(ArgumentError("kmax must be at least kmin"))
    (k < kmin || (!isnothing(kmax) && k > kmax)) && return 0.0
    return exp(_qexp_logpk(k, q, λ, kmin, kmax))
end

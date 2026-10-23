# ---------------------------------------------------------------------------
# Degree sequences
# ---------------------------------------------------------------------------

"""
    sample_qexp_degrees(N, q, λ; kmin=2, rng)

Draw `N` integer degrees from the q-exponential of Sampaio Filho et al.,
PRR 5, 033088 (2023), Eq. (1),

    P_q(x) = (2 - q) λ [1 + (q - 1) λ x]^{-1/(q-1)},   x ≥ 0,

by inverse transform, and set k = floor(x). Draws with k < kmin are redrawn,
so P(k) is the q-exponential truncated to k ≥ kmin. Degrees are capped at
N - 1, and if the total degree is odd one node is redrawn until it is even.
q = 1 gives the exponential limit, P_1(x) = λ exp(-λ x).
"""
function sample_qexp_degrees(N::Integer, q::Real, λ::Real; kmin::Integer=2,
    rng::AbstractRNG=Random.default_rng())
    @assert 1 <= q < 2 "need 1 ≤ q < 2 for a normalizable q-exponential"
    a = (q - 1) / (q - 2)   # exponent of the inverse CDF
    draw() = begin
        while true
            u = rand(rng)
            x = q == 1 ? -log(1 - u) / λ : ((1 - u)^a - 1) / ((q - 1) * λ)
            k = x >= N - 1 ? N - 1 : floor(Int, x)
            k >= kmin && return k
        end
    end
    degs = [draw() for _ in 1:N]
    while isodd(sum(degs))
        degs[rand(rng, 1:N)] = draw()
    end
    return degs
end

"""
    sample_qexp_degrees_discrete(N, q, κ; kmin=2, kmax=N-1, rng)

Draw `N` integer degrees k ≥ kmin from the discrete q-exponential

    P(k) ∝ e_q(-k/κ) = [1 + (q - 1) k/κ]^{-1/(q-1)},   kmin ≤ k ≤ kmax,

the form fitted to log-binned degree distributions (Brito et al., Sci. Rep. 6,
27992 (2016)), by inverse transform on its cumulative sum. With `kmax` < N - 1
the distribution is truncated at kmax (and renormalized), e.g. at the largest
degree of the networks it should reproduce. Unlike
`sample_qexp_degrees`, which floors a continuous variable, P(k) is exactly this
function of (q, κ) at every integer k. If the total degree is odd one node is
redrawn until it is even. q = 1 gives P(k) ∝ exp(-k/κ).
"""
function sample_qexp_degrees_discrete(N::Integer, q::Real, κ::Real; kmin::Integer=2,
    kmax::Integer=N - 1, rng::AbstractRNG=Random.default_rng())
    @assert q >= 1 && κ > 0 "need q ≥ 1 and κ > 0"
    @assert kmin <= kmax <= N - 1 "need kmin ≤ kmax ≤ N - 1"
    w = [q == 1 ? exp(-k / κ) : (1 + (q - 1) * k / κ)^(-1 / (q - 1)) for k in kmin:kmax]
    cdf = cumsum(w) ./ sum(w)
    draw() = kmin - 1 + min(searchsortedfirst(cdf, rand(rng)), length(cdf))
    degs = [draw() for _ in 1:N]
    while isodd(sum(degs))
        degs[rand(rng, 1:N)] = draw()
    end
    return degs
end

"""
    qexp_pk(k, q, λ; kmin=2)

Probability of the integer degree `k` produced by `sample_qexp_degrees`,
P(k) = [F(k+1) - F(k)] / [1 - F(kmin)], with F the CDF of Eq. (1).
"""
function qexp_pk(k, q, λ; kmin=2)
    S(x) = q == 1 ? exp(-λ * x) : (1 + (q - 1) * λ * x)^((q - 2) / (q - 1))   # 1 - F(x)
    return (S(k) - S(k + 1)) / S(kmin)
end

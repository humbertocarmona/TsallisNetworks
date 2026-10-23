# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

"""
    robustness(g; attack=:random, rng)
    robustness(g, order)

Robustness of `g` to the removal of its nodes: by random failure
(`attack=:random`, a random permutation from `rng`), by adaptive targeted
attack (`attack=:targeted`, `attack_order`), or in a given `order`.
Returns a named tuple with

  - `f`, `S`, `κ`: for every fraction f = 0, 1/N, ..., 1 of removed nodes, the
    fraction S of the nodes in the largest connected component and
    κ = ⟨k²⟩/⟨k⟩ of that component (`percolation`);
  - `fc_mr`: the smallest f with κ ≤ 2 (Molloy-Reed criterion,
    `molloy_reed_fc`; exact for uncorrelated networks only);
  - `fc_jump`: the f of the largest drop of S in a single removal (no
    assumption on the correlations);
  - `Rindex`: the robustness index of Schneider et al., PNAS 108, 3838
    (2011), R = (1/N) Σ_{Q=1}^{N} S(Q/N), between 0 and 1/2.
"""
function robustness(g::SimpleGraph; attack=:random, rng::AbstractRNG=Random.default_rng())
    a = Symbol(attack)
    order = a === :random ? randperm(rng, nv(g)) :
            a === :targeted ? attack_order(g; rng=rng) :
            throw(ArgumentError("attack must be :random or :targeted, got $attack"))
    return robustness(g, order)
end

function robustness(g::SimpleGraph, order::AbstractVector{<:Integer})
    N = nv(g)
    S, κ = percolation(g, order)    # S[i], κ[i]: i - 1 nodes removed
    ijump = argmax(S[1:end-1] .- S[2:end])
    return (f=collect(0:N) ./ N, S=S, κ=κ, fc_mr=molloy_reed_fc(κ),
        fc_jump=ijump / N, Rindex=sum(@view S[2:end]) / N)
end

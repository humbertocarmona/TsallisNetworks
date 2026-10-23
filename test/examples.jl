# Examples: generate networks with the C-R and P-R models of Ochiai & Nacher,
# Physica A 388, 4887 (2009), and compute their observables. Not part of the
# test suite (runtests.jl); run it from the package folder with
#
#   julia --project=. -t 4 test/examples.jl
#
# (-t: mean_shortest_path runs its breadth-first searches on all threads).

using TsallisNetworks                       # expq, lnq, save_bz2_csv, load_bz2_csv
using TsallisNetworks.CopyRandom            # copy_random_graph
using TsallisNetworks.PreferentialRandom    # preferential_random_graph, pr_pk, pr_qexp
using TsallisNetworks.Observables           # mean_shortest_path, clustering, ...
using DataFrames
using Graphs
using Random
using Statistics

N = 10_000

# ---------------------------------------------------------------------------
# 1. Generate the networks
# ---------------------------------------------------------------------------

# C-R model: with probability α copy a random node and its edges, otherwise add
# a node linked to m0 random nodes. A seeded rng makes the network reproducible.
g_cr = copy_random_graph(N, 0.4; m0=2, rng=Xoshiro(1))

# P-R model: each new node brings one edge, attached preferentially (∝ degree)
# with probability α, otherwise to a random node. It grows a tree on the seed
# clique of n0 nodes.
α = 0.6
g_pr = preferential_random_graph(N, α; rng=Xoshiro(1))

for (name, g) in (("C-R", g_cr), ("P-R", g_pr))
    println("$name: N = $(nv(g)), M = $(ne(g)), <k> = $(round(2ne(g) / nv(g); digits=2)), ",
        "kmax = $(maximum(degree(g))), connected = $(is_connected(g))")
end

# ---------------------------------------------------------------------------
# 2. Degree distribution of the P-R model against its q-exponential
# ---------------------------------------------------------------------------

# exact P(k) of the rate equation, and the q-exponential e_q(-k/κ) of its tail
(; q, κ) = pr_qexp(α)
pk = pr_pk(α, 10)
qexp = [expq(-k / κ, q) for k in 1:10^6]
qexp ./= sum(qexp)
degs = degree(g_pr)
println("\nP-R, α = $α: q = $(round(q; digits=4)), κ = $(round(κ; digits=4))")
println(DataFrame(k=1:10, simulation=[mean(degs .== k) for k in 1:10],
    rate_equation=pk, qexp=qexp[1:10]))

# ---------------------------------------------------------------------------
# 3. Observables of one network
# ---------------------------------------------------------------------------

g = g_cr

# average shortest path length <ℓ>: exact (BFS from every node), or estimated
# from a random sample of sources; giant=true restricts it to the largest
# connected component
ell = mean_shortest_path(g)
ell_est = mean_shortest_path(g; nsources=500, rng=Xoshiro(2))
println("\nC-R: <ℓ> = $(round(ell; digits=3)), from 500 sources: $(round(ell_est; digits=3))")

# clustering: mean local clustering C and transitivity T, and C(k)
(; C, T) = clustering(g)
ck = clustering_spectrum(g)                  # named tuple of vectors (k, Ck, n)
println("C = $(round(C; digits=4)), T = $(round(T; digits=4))")
println("C(k), first degrees: ", first(DataFrame(ck), 5))

# degree correlations: k_nn(k), assortativity R, Spearman ρ, μ = <k²>/<k>
dc = degree_correlations(g)
println("R = $(round(dc.R; digits=4)), ρ = $(round(dc.rho; digits=4)), μ = $(round(dc.mu; digits=2))")
println("k_nn(k), first degrees: ", first(DataFrame(dc.knn), 5))

# k-core: highest core index k_h and its number of nodes M_h
kh, Mh = kcore_stats(g)
println("k_h = $kh, M_h = $Mh")

# robustness: S(f) of the largest component as a fraction f of the nodes is
# removed, at random or hubs first (degrees recomputed after each removal)
rand_fail = robustness(g; attack=:random, rng=Xoshiro(3))
attack = robustness(g; attack=:targeted, rng=Xoshiro(3))
for (name, r) in (("random failure", rand_fail), ("targeted attack", attack))
    println("$name: f_c (Molloy-Reed) = $(round(r.fc_mr; digits=3)), ",
        "f_c (largest jump) = $(round(r.fc_jump; digits=3)), R index = $(round(r.Rindex; digits=3))")
end
# S at a few fractions f of removed nodes (r.S[i] has i - 1 nodes removed)
fs = [0.0, 0.1, 0.2, 0.5]
println(DataFrame(f=fs, S_random=[rand_fail.S[round(Int, f * N)+1] for f in fs],
    S_targeted=[attack.S[round(Int, f * N)+1] for f in fs]))

# all scalar observables at once (⟨ℓ⟩ here on the largest component, 200 sources)
println("\nobservables: ", observables(g; rng=Xoshiro(4)))

# ---------------------------------------------------------------------------
# 4. Ensemble: observables over several realizations, both models
# ---------------------------------------------------------------------------

rows = map(Iterators.product(1:5, ("CR", "PR"))) do (seed, model)
    rng = Xoshiro(100 + seed)
    g = model == "CR" ? copy_random_graph(N, 0.4; m0=2, rng=rng) :
        preferential_random_graph(N, α; rng=rng)
    (; C, T) = clustering(g)
    r = robustness(g; attack=:targeted, rng=rng)
    (model=model, seed=seed, ell=mean_shortest_path(g; nsources=500, rng=rng),
        C=C, T=T, R=degree_correlations(g).R, fc_attack=r.fc_jump, Rindex=r.Rindex)
end
ens = DataFrame(vec(rows))
println("\nper realization:\n", ens)
println("\nmean over realizations:\n",
    combine(groupby(ens, :model), [:ell, :C, :T, :R, :fc_attack, :Rindex] .=> mean; renamecols=false))

# save and read back as a bz2-compressed CSV (strings, not Symbols, survive the
# round trip)
path = tempname() * ".csv.bz2"
save_bz2_csv(path, ens)
@assert load_bz2_csv(path) == ens
println("\nsaved and read back: $path")

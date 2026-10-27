# Examples: generate C-R and P-R networks of Ochiai & Nacher, Physica A 388,
# 4887 (2009), and edge-recruitment networks; compute their observables and
# illustrate productivity-driven collaborator counts. Not part of the
# test suite (runtests.jl); run it from the package folder with
#
#   julia --project=. -t 4 test/examples.jl
#
# (-t: mean_shortest_path runs its breadth-first searches on all threads).

using TsallisNetworks                       # expq, lnq, save_bz2_csv, load_bz2_csv
using TsallisNetworks.CopyRandom            # copy_random_graph
using TsallisNetworks.PreferentialRandom    # preferential_random_graph, pr_pk, pr_qexp
using TsallisNetworks.EdgeRecruitment       # physical multigraph and degree predictions
using TsallisNetworks.Observables           # mean_shortest_path, clustering, ...
using DataFrames
using Graphs
using Random
using Statistics

N = 10_000
# %% --------------------------------------------------------------------------------
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
qexp = [expq(-k / κ, q) for k in 1:(10^6)]
qexp ./= sum(qexp)
degs = degree(g_pr)
println("\nP-R, α = $α: q = $(round(q; digits=4)), κ = $(round(κ; digits=4))")
println(DataFrame(k=1:10, simulation=[mean(degs .== k) for k in 1:10],
    rate_equation=pk, qexp=qexp[1:10]))
# Multi-edge P-R extension: m0 distinct targets per arrival, with frozen weights.
# Repeat the whole preferential/uniform mixture after duplicate-target rejection.
m_pr = 2
α_pr_multi = 0.6
g_pr_multi = preferential_random_graph(N, α_pr_multi; m0=m_pr, rng=Xoshiro(2))
deg_pr_multi = degree(g_pr_multi)
pk_pr_multi = pr_pk(α_pr_multi, 20; m0=m_pr)
parameters_pr_multi = pr_qexp(α_pr_multi; m0=m_pr, approximation=:continuum)
println("\nMulti-edge P-R: m0=$m_pr, continuum parameters=$parameters_pr_multi")
println(DataFrame(k=m_pr:20,
    simulation=[mean(deg_pr_multi .== k) for k in m_pr:20],
    stationary_prediction=pk_pr_multi[m_pr:20]))
# This recurrence uses the large-N approximation to distinct-target inclusion
# probabilities. The continuum q-exponential makes an additional degree approximation.

# %% --------------------------------------------------------------------------------
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
println(DataFrame(f=fs, S_random=[rand_fail.S[round(Int, f*N)+1] for f in fs],
    S_targeted=[attack.S[round(Int, f*N)+1] for f in fs]))

# all scalar observables at once (⟨ℓ⟩ here on the largest component, 200 sources)
println("\nobservables: ", observables(g; rng=Xoshiro(4)))
# %% --------------------------------------------------------------------------------
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
# %% --------------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 5. Recruitment through edges: multigraph degrees and simple projection
# ---------------------------------------------------------------------------

# The L=0 edge-attachment rule is from Dorogovtsev, Mendes & Samukhin,
# Phys. Rev. E 63, 062101 (2001), doi:10.1103/PhysRevE.63.062101.
# L>0 adds the virtual-loop extension described in the generator's comments.
# Each node has L virtual recruitment loops, excluded from physical degree.
# Selecting a physical edge forms a triangle; selecting a virtual loop adds
# two parallel physical edges to its owner. Every newcomer has degree two.
L = 1
N=1_000_000
g_er = edge_recruitment_graph(N; L=L, n0=3, rng=Xoshiro(5))
physical_degrees = degree(g_er)
println("\nEdge recruitment, L = $L: N = $(nv(g_er)), physical edges = $(ne(g_er)), ",
    "<k> = $(round(mean(physical_degrees); digits=3))")

# Compute the complete empirical degree PMF, including zero-count degrees.
# Predictions concern physical multiplicities, not the simple projection.
er_empirical = recruitment_degree_distribution(g_er)
er_parameters = recruitment_qexp(L)
er_kmax = last(er_empirical.k)
er_pk = recruitment_pk(L, er_kmax)
er_continuum(k) = (L + 2) / (2L + 2) * ((k + 2L) / (2L + 2))^(-(L + 3))
# Integrate the normalized continuum density over [k,k+1) for a PMF
# comparison; density values at integer k are not discrete probabilities.
er_survival(k) = ((k + 2L) / (2L + 2))^(-(L + 2))
er_distribution = DataFrame(k=er_empirical.k, count=er_empirical.n,
    simulation=er_empirical.pk,
    stationary_discrete=vcat(0.0, er_pk),
    continuum = [k < 2 ? 0.0 : er_continuum(k)
        for k in er_empirical.k],
    continuum_bin_mass=[k < 2 ? 0.0 : er_survival(k) - er_survival(k + 1)
        for k in er_empirical.k])
er_distribution.residual = er_distribution.simulation .- er_distribution.stationary_discrete
println("Continuum parameters: ", er_parameters)  # q = 1.25, κ = 0.5
println("Physical degree distribution (first 20 degrees; full table: er_distribution):")
println(first(er_distribution, 20))
# %% --------------------------------------------------------------------------------
# Keep the predicted probability beyond the observed maximum: renormalizing
# at that maximum would conceal part of the finite-size discrepancy.
er_predicted_tail = max(0.0, 1 - sum(er_pk))
er_total_variation = (sum(abs, er_distribution.residual) + er_predicted_tail) / 2
println("Observed maximum degree = $er_kmax, predicted mass above it = $er_predicted_tail")
println("Total variation from the limiting discrete law = $er_total_variation ",
    "(descriptive finite-size comparison, not a goodness-of-fit p-value)")

# Existing observables operate on simple graphs. Projection collapses parallel
# edges and changes degrees: its statistics describe the projection, whereas
# the predictions above describe the physical multigraph.
h_er = simple_projection(g_er)
println("Simple projection: edges = $(ne(h_er)), collapsed parallel edges = ",
    ne(g_er) - ne(h_er), ", connected = $(is_connected(h_er))")
println("Projected clustering: ", clustering(h_er))
println("Projected <ℓ> (500 sources): ",
    mean_shortest_path(h_er; nsources=500, rng=Xoshiro(6)))

# L = 0 has no virtual-loop events and therefore no parallel edges.
# Its continuum law is the pure power law 8k^-3: do not evaluate expq(-k/κ,q)
# with the returned κ = 0.
g_er0 = edge_recruitment_graph(N; L=0, rng=Xoshiro(7))
@assert ne(simple_projection(g_er0)) == ne(g_er0)
println("L = 0 continuum parameters (pure power law): ", recruitment_qexp(0))

# ---------------------------------------------------------------------------
# 6. Productivity-driven collaborator counts (ego-level toy model, not a graph)
# ---------------------------------------------------------------------------
using TsallisNetworks.ProductivityCollaboration

# Motivation: Frota et al., "Causal asymmetry suggests productivity underlies
# scientific collaboration", PNAS Nexus (2026), doi:10.1093/pnasnexus/pgag360.
# Their conditional statistics motivate the geometric–lognormal mixture.
# The Yule exposure mechanism below is a proposed explanation, not a mechanism
# established by that paper. These parameter values illustrate its reported
# scaling; no empirical data are fitted or loaded by this example.
collab_settings = (b=0.33, beta=1.3, logmean=4.22, logsigma=0.75)
collab_population = productivity_collaboration(N; collab_settings..., rng=Xoshiro(80))
println("\nProductivity-driven population: ", length(collab_population.collaborators), " egos")
println("Mean collaborator count = ", mean(collab_population.collaborators),
    "; mean specified conditional mean = ", mean(collab_population.conditional_mean))

# Compare the sampled marginal with deterministic integration over productivity.
# These are probability masses, with no renormalization to the displayed range.
collab_ks = collect(1:20)
collab_theory = collaboration_mixture_pmf(collab_ks; collab_settings...)
println(DataFrame(C=collab_ks,
    simulation=[mean(collab_population.collaborators .== k) for k in collab_ks],
    mixture_prediction=collab_theory))

# At fixed productivity, verify the geometric mean and variance separately.
fixed_productivity = 50.0
collab_fixed = productivity_collaboration(fill(fixed_productivity, 50_000);
    b=collab_settings.b, beta=collab_settings.beta, rng=Xoshiro(81))
collab_mu = collaboration_mean(fixed_productivity; b=collab_settings.b, beta=collab_settings.beta)
println("Fixed A=$fixed_productivity: sample mean = ", mean(collab_fixed.collaborators),
    ", prediction = ", collab_mu, "; sample variance = ", var(collab_fixed.collaborators),
    ", prediction = ", collab_mu * (collab_mu - 1))

# One coupled trajectory: existing collaborators create further opportunities
# at rate C in exposure s(A)=log(max(1,b*A^beta)). Productivity drives the clock.
# These A values are continuous covariates/observation points, not paper events.
collab_path = collaboration_trajectory([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
    b=collab_settings.b, beta=collab_settings.beta, rng=Xoshiro(82))
println("One exposure-time trajectory:\n", DataFrame(collab_path))
# No adjacency matrix or reciprocal collaborator identities are generated.

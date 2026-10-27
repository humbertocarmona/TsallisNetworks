# TsallisNetworks

Growth and null models for networks with q-exponential (Tsallis) degree distributions.

| Submodule | Contents |
|---|---|
| `TsallisNetworks.CopyRandom` | copy/random growth model, `copy_random_graph` |
| `TsallisNetworks.ConfigurationModel` | q-exponential degree sequences and configuration model (`stub_matching`, `rewire`); re-exports the observables it used to hold |
| `TsallisNetworks.PreferentialRandom` | P-R model of Ochiai & Nacher, Physica A 388, 4887 (2009): `preferential_random_graph`; exact P(k) of its rate equation, `pr_pk`, and its q-exponential parameters, `pr_qexp` |
| `TsallisNetworks.EdgeRecruitment` | physical multigraph growth with virtual recruitment loops: `edge_recruitment_graph`, stationary discrete probabilities `recruitment_pk`, and continuum parameters `recruitment_qexp` |
| `TsallisNetworks.ProductivityCollaboration` | productivity-conditioned geometric collaborator counts, exact Yule trajectories, and a geometric–lognormal mixture (ego-level model) |
| `TsallisNetworks.Observables` | observables of any network: ⟨ℓ⟩ (`mean_shortest_path`), clustering (`clustering`, `clustering_spectrum`), degree correlations (`knn`, `degree_correlations`: k_nn(k), R, ρ, μ), k-core (`kcore_stats`), robustness (`robustness`: S(f), κ(f), f_c, Schneider's R), `observables` |
| `TsallisNetworks.SpatialPreferentialAttachment` | spatial preferential attachment model, `create_spa_graph`; `plotgraph` is available once Makie (or a Makie backend) is loaded |

`expq`, `lnq`, `save_bz2_csv` and `load_bz2_csv` are exported from the top level and re-exported by the submodules that use them.

```julia
using TsallisNetworks.SpatialPreferentialAttachment
g, pos, w = create_spa_graph(10_000, 2; d=2, αA=2.0)

using TsallisNetworks.Observables
mean_shortest_path(g; nsources=1000)        # ⟨ℓ⟩, threaded
clustering(g)                               # (C, T)
degree_correlations(g).R                    # assortativity
robustness(g; attack=:targeted).fc_jump     # critical fraction
```

## Sampling and model conventions

- `sample_qexp_degrees` floors a continuous draw conditional on `x ≥ kmin`
  and caps it at `N-1`. Use `qexp_pk(k, q, λ; kmin, kmax=N-1)` to include
  the probability accumulated at the cap. Omitting `kmax` describes the
  unbounded distribution. Here `λ` is an inverse scale, unlike `κ` in the
  discrete sampler.
- `sample_qexp_degrees_discrete` samples point weights `exp_q(-k/κ)` on
  `kmin:kmax`, normalized on that finite support. Both samplers repair an
  odd total by redrawing one randomly chosen degree conditional on opposite
  parity. This is a sequence repair, not independent sampling conditioned on
  the total. Invalid or impossible supports raise `ArgumentError`.
- `stub_matching` constructs a simple graph and may discard stubs. Inspect
  the returned remainder and realized degrees. `rewire` preserves degrees
  exactly but does not preserve connectivity or guarantee mixing after a
  fixed number of attempts; its acceptance fraction is zero when no attempts
  are made.
- SPA defaults to `cm=true`: arrival displacements are measured from the
  current center of mass. Set `cm=false` for a fixed-origin arrival law.
  Its default radial cutoff is `rmax=1000.0`; set it explicitly when comparing
  with a manuscript or another implementation.
- C–R defaults to `link_parent=false`, matching Ochiai and Nacher.
- P–R defaults to the original one-edge model. Set `m0>1` for our fixed-birth-degree
  extension: freeze pre-arrival weights, sample distinct targets, and redraw the
  whole preferential/uniform mixture when rejecting duplicates. Its mean degree
  approaches `2m0`. `pr_pk(α, kmax; m0)` evaluates the asymptotic stationary
  recurrence; for multiple targets it neglects finite-size inclusion corrections.
  `pr_qexp(α; m0, approximation=:continuum)` gives
  `q=1+α/(α+2)` and `κ=2(1-α)m0/(α+2)`. The default `:discrete` mode instead
  gives the large-degree shifted-power-law scale `κ=(2(1-α)m0+1)/(α+2)`
  for α>0, and the exact geometric scale `1/log1p(1/m0)` at α=0.
  At α=1 the continuum is a pure power law (`κ=0`). Neither mode asserts an
  exact discrete q-exponential.

```julia
using TsallisNetworks.PreferentialRandom
using Random

g = preferential_random_graph(100_000, 0.6; m0=3, rng=Xoshiro(1))
pk = pr_pk(0.6, 100; m0=3)  # entries are P(k), k=1:100; zero below 3
parameters = pr_qexp(0.6; m0=3, approximation=:continuum)
```

The stable conditional samplers change the random-number stream relative to
the former rejection samplers; the same seed need not reproduce old degree
sequences.

## Recruitment through edges

```julia
using TsallisNetworks.EdgeRecruitment
using Graphs, Random

g = edge_recruitment_graph(10_000; L=1, n0=3, rng=Xoshiro(1))
nv(g), ne(g), degree(g)             # physical degrees count parallel edges
edge_multiplicity(g, 1, 2)
p = recruitment_pk(1, 100)          # p[k], discrete law; p[1] = 0
dist = recruitment_degree_distribution(g)        # (k, pk, n): degree values, probabilities, counts
q, κ = recruitment_qexp(1)          # q = 1.25, κ = 0.5; continuum only

h = simple_projection(g)           # collapse parallel edges explicitly
# Existing Observables functions accept h; its degrees differ from g's.
```

Start with a physical clique. Each vertex has `L` virtual loops, excluded from
physical edges and degrees. Uniformly select a physical edge (including each
parallel copy) or a virtual loop. Attach a newcomer to both endpoints and give
it `L` virtual loops. Physical-edge events close triangles; virtual-loop events
create two parallel edges to their owner. The selected object remains.

The generator stores physical edges individually and samples virtual loops
implicitly. For `N` vertices it has `n0*(n0-1)/2 + 2*(N-n0)` physical edges.
`L=0` is supported and gives ordinary edge-based triangle growth.

`RecruitmentGraph` deliberately does not subtype `Graphs.AbstractGraph`:
simple-graph algorithms could otherwise silently misinterpret multiplicity.
It supports `Graphs.nv`, `ne`, `vertices`, `edges`, and `degree`; `edges` yields
every physical edge separately. Use `simple_projection` for simple-graph
metrics, explicitly recognizing the changed degree sequence. Virtual loops
are never included in either representation.

The limiting discrete recurrence is
`(k+2L+2)n[k] = (k-1)n[k-1] + L*n[k-2] + (L+2)δ[k,2]`.
`recruitment_pk` solves it without renormalizing at the requested cutoff.
The continuum density on `k≥2` is
`(L+2)/(2+2L) * ((k+2L)/(2+2L))^(-(L+3))`, with
`q=1+1/(L+3)` and `κ=2L/(L+3)`. These are continuum predictions, not an exact
discrete q-exponential. At `L=0` the continuum density is the pure power law
`8k^-3`; the returned zero scale must not be used directly in `exp_q(-k/κ)`.

## Productivity-driven collaborator counts

```julia
using TsallisNetworks.ProductivityCollaboration
using Random

# Fixed measured productivities, with independent conditional counts:
pop = productivity_collaboration([10, 20, 50]; b=0.33, beta=1.3, rng=Xoshiro(1))
μ = collaboration_mean(50; b=0.33, beta=1.3)
p = collaboration_pmf(3, 50; b=0.33, beta=1.3)

# Lognormal productivity is a continuous covariate, not rounded paper counts:
pop = productivity_collaboration(10_000; logmean=4.22, logsigma=0.75,
    b=0.33, beta=1.3, rng=Xoshiro(2))
ks = collect(1:100)
pred = collaboration_mixture_pmf(ks; logmean=4.22, logsigma=0.75, b=0.33, beta=1.3)

# Exact coupled Yule path observed as productivity increases:
path = collaboration_trajectory([1, 2, 5, 10, 50]; b=0.33, beta=1.3, rng=Xoshiro(3))
```

The `ProductivityCollaboration` submodule models ego-level counts, not a global
graph. It is motivated by Frota et al., *Causal asymmetry suggests productivity
underlies scientific collaboration*, PNAS Nexus (2026),
[doi:10.1093/pnasnexus/pgag360](https://doi.org/10.1093/pnasnexus/pgag360).
The exposure-time Yule mechanism is a proposed toy explanation, not a mechanism
validated by that empirical study. The package's default parameters are neutral
illustrations; the values above are explicit example settings.

The conditional mean is `μ(A)=max(1,b*A^beta)`. Starting at C=1 and exposure s=0,
the exact pure-birth process has transition rate C for C→C+1. Observing it at
`s=log(μ(A))` gives the geometric law on C=1,2,..., with mean μ and variance
μ(μ−1). The low-A plateau is explicit; this model does not include C=0.
The clock encodes the specified mean relation rather than predicting beta.

`sample_collaborators` samples an endpoint in O(1) time.
`productivity_collaboration` samples independent conditional counts at supplied
productivities, or generates lognormal productivities first. Natural-log
parameters are used. `collaboration_trajectory` generates exponential holding
times and couples observations along one increasing productivity path; its cost
scales with the number of realized births. Both return named tuples suitable
for `DataFrame` construction. Virtual exposure is not calendar time.

`collaboration_mixture_pmf` integrates over the standard-normal productivity
coordinate by Simpson quadrature, splitting at the mean-one boundary. The
normal range and finite displayed count range are not renormalized; increase
`intervals` and `zmax` to check convergence. A general geometric–lognormal
mixture is not an exact q-exponential.

Collaborator identities, reciprocity, finite population limits, publication
teams, and repeated coauthor events are outside this count model. Do not pass
its output to graph observables or assume that the counts form a graphical
degree sequence. Network construction requires an additional mechanism.

# TsallisNetworks

Growth and null models for networks with q-exponential (Tsallis) degree distributions.

| Submodule | Contents |
|---|---|
| `TsallisNetworks.CopyRandom` | copy/random growth model, `copy_random_graph` |
| `TsallisNetworks.ConfigurationModel` | q-exponential degree sequences and configuration model (`stub_matching`, `rewire`); re-exports the observables it used to hold |
| `TsallisNetworks.PreferentialRandom` | P-R model of Ochiai & Nacher, Physica A 388, 4887 (2009): `preferential_random_graph`; exact P(k) of its rate equation, `pr_pk`, and its q-exponential parameters, `pr_qexp` |
| `TsallisNetworks.EdgeRecruitment` | physical multigraph growth with virtual recruitment loops: `edge_recruitment_graph`, stationary discrete probabilities `recruitment_pk`, and continuum parameters `recruitment_qexp` |
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
- C–R defaults to `link_parent=false`, matching Ochiai and Nacher. P–R adds
  one edge per newcomer. `pr_pk` gives the discrete stationary recurrence;
  `pr_qexp` returns the scale of its large-degree shifted-power-law
  approximation, `κ=(3-2α)/(α+2)`, rather than the unshifted continuum scale
  `κ=2(1-α)/(α+2)`. Neither asserts an exact discrete q-exponential.

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

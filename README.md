# TsallisNetworks

Growth and null models for networks with q-exponential (Tsallis) degree distributions.

| Submodule | Contents |
|---|---|
| `TsallisNetworks.CopyRandom` | copy/random growth model, `copy_random_graph` |
| `TsallisNetworks.ConfigurationModel` | q-exponential degree sequences and configuration model (`stub_matching`, `rewire`); re-exports the observables it used to hold |
| `TsallisNetworks.PreferentialRandom` | P-R model of Ochiai & Nacher, Physica A 388, 4887 (2009): `preferential_random_graph`; exact P(k) of its rate equation, `pr_pk`, and its q-exponential parameters, `pr_qexp` |
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

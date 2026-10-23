module ConfigurationModel

using Graphs
using Random
using Statistics
using StatsBase

import ..TsallisNetworks: load_bz2_csv, save_bz2_csv
# observables used to live here; re-exported so that old scripts keep working
import ..Observables: attack_order, kcore_stats, knn, mean_shortest_path,
    molloy_reed_fc, observables, percolation

include("degree_sequences.jl")
include("null_models.jl")

export attack_order,
    kcore_stats,
    knn,
    load_bz2_csv,
    mean_shortest_path,
    molloy_reed_fc,
    observables,
    percolation,
    qexp_pk,
    rewire,
    sample_qexp_degrees,
    sample_qexp_degrees_discrete,
    save_bz2_csv,
    stub_matching

end

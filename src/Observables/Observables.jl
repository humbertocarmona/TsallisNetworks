module Observables

using Graphs
using Random
using Statistics
using StatsBase

include("observables.jl")
include("clustering.jl")
include("correlations.jl")
include("robustness.jl")

export attack_order,
    clustering,
    clustering_spectrum,
    degree_correlations,
    kcore_stats,
    knn,
    knn_table,
    mean_shortest_path,
    molloy_reed_fc,
    observables,
    percolation,
    robustness

end

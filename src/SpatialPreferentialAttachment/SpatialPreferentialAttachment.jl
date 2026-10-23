module SpatialPreferentialAttachment
using DataFrames, DataFramesMeta
using Graphs
using Memento
using Random
using SparseArrays

import LinearAlgebra: norm
import Statistics: mean
import StatsBase: sample, Weights
import ..TsallisNetworks: expq, lnq, load_bz2_csv, save_bz2_csv

include("utils.jl")
include("circular_layout.jl")
include("creategraph.jl")
include("attach_list.jl")
include("add_node.jl")
include("create_spa_graph.jl")
include("degree_distribution.jl")

"""
    plotgraph(nodes_df::DataFrame, edges_df::DataFrame; kwargs...)

    Plots a graph given dataframes for nodes and edges. Needs Makie (or a
    backend such as CairoMakie) to be loaded; the method lives in the
    `TsallisNetworksMakieExt` extension, where the keywords are documented.
"""
function plotgraph end

const logger = getlogger(@__MODULE__)
function __init__()
    Memento.config!("debug"; fmt="{level}: {msg}")
    setlevel!(logger, "debug")
    Memento.register(logger)
end


export add_node!,
    circular,
    creategraph,
    create_spa_graph,
    degree_distribution,
    expq,
    lnq,
    load_bz2_csv,
    map_to_interval,
    plotgraph,
    position_layout,
    save_bz2_csv
end

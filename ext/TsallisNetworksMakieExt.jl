module TsallisNetworksMakieExt

using DataFrames, DataFramesMeta
using Makie
using TsallisNetworks.SpatialPreferentialAttachment:
    circular, map_to_interval, position_layout
import TsallisNetworks.SpatialPreferentialAttachment: plotgraph
import Statistics: mean

"""
    plotgraph(
        nodes_df::DataFrame,
        edges_df::DataFrame;
        nodes_sortby=nothing,
        node_color_col=(:gray21, 0.4),
        node_strokecolor=:transparent,
        node_size=nothing,
        edge_alpha=1.0,
        edge_width=1.0,
        edge_color_col=:len,
        ecmin=nothing,
        ecmax=nothing,
        n_curve_points::Int=30,
        figsize::Tuple{Int,Int}=(600, 600),
        title::AbstractString="",
        layout=circular,
        ax=nothing
    )

    Plots a graph given dataframes for nodes and edges.

    # Arguments
    - `nodes_df::DataFrame`: DataFrame containing node information. Must have `:id` and `:name` columns.
    - `edges_df::DataFrame`: DataFrame containing edge information. Must have `:source`, `:target`, and `:len` columns.

    # Keywords
    - `nodes_sortby`: Column(s) to sort nodes by for layout calculation.
    - `node_color_col`: Color or tuple (color, alpha) for nodes. Can also be a column name from `nodes_df`.
    - `node_strokecolor`: Stroke color for nodes.
    - `node_size`: Size of nodes. Can be a scalar, a vector, or a column name from `nodes_df`.
    - `edge_alpha`: Transparency of edges.
    - `edge_width`: Width of edges.
    - `edge_color_col`: Column from `edges_df` to use for coloring edges.
    - `ecmin`: Minimum value for edge color scaling. If `nothing`, uses `minimum(edges_df[!, edge_color_col])`.
    - `ecmax`: Maximum value for edge color scaling. If `nothing`, uses `maximum(edges_df[!, edge_color_col])`.
    - `n_curve_points::Int`: Number of points to use for drawing curved edges.
    - `figsize::Tuple{Int,Int}`: Figure size in pixels.
    - `title::AbstractString`: Title of the plot.
    - `layout`: Function to compute node positions (e.g., `circular`, `spring`).
    - `ax`: A `Makie.Axis` object to plot on. If `nothing`, a new figure and axis are created.

    # Returns
    - `Makie.Figure`: The generated Makie figure.
"""
function plotgraph(
    nodes_df::DataFrame,
    edges_df::DataFrame;
    nodes_sortby=nothing, # list of columns
    node_color_col=(:gray21, 0.4),
    node_strokecolor=:transparent,
    node_size=nothing,
    edge_alpha=1.0,
    edge_width=1.0,
    edge_color_col=:len,
    ecmin=nothing,
    ecmax=nothing,
    n_curve_points::Int=30,
    figsize::Tuple{Int,Int}=(600, 600),
    title::AbstractString="",
    layout=circular,
    ax=nothing)

    pos = layout(nodes_df; sortby=nodes_sortby)
    a = 1 / 25
    b = 1 / 5
    if node_size isa AbstractVector
        markersize = node_size
        markersize = map_to_interval(markersize, a=a, b=b)
    elseif node_size isa Union{Symbol,String}
        @assert String(node_size) ∈ names(nodes_df)
        markersize = nodes_df[!, node_size]
        markersize = map_to_interval(markersize, a=a, b=b)
    elseif node_size isa Float32
        markersize = fill(node_size, nrow(nodes_df))
    else
        markersize = a
    end

    edge_color_scale = cgrad(:jet, 256, rev=false)
    edge_color_scale_len = length(edge_color_scale)

    if isnothing(ax)
        fig = Figure(; size=figsize, figure_padding=2, backgroundcolor=:white)
        ax = Axis(fig[1, 1], title=title, aspect=DataAspect(), backgroundcolor=:white)
    else
        fig = current_figure()
    end

    ec = edges_df[!, edge_color_col]
    if isnothing(ecmin)
        ecmin = minimum(ec)
    end
    if isnothing(ecmax)
        ecmax = maximum(ec)
    end
    println("cmin = $ecmin, cmax = $ecmax")

    edge_colors = map_to_interval(log10.(ec), log10(ecmin), log10(ecmax); a=1, b=edge_color_scale_len)
    edge_colors = round.(Int, edge_colors)
    edges_df[!, :ec] = edge_colors

    # draw edges (first layer), sorted by edge_color_col
    edf = copy(edges_df)

    edf = @orderby(edf, $edge_color_col)
    
    for row in eachrow(edf)
        s = row.source
        t = row.target
        x0, y0 = pos.x[s], pos.y[s]
        x1, y1 = pos.x[t], pos.y[t]
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        if layout == circular
            # Bézier control point at mid-angle, smaller radius (near center)
            ang0 = atan(y0, x0)
            ang1 = atan(y1, x1)
            angm = (ang0 + ang1) / 2

            r_ctrl = 0       # distance of the control point from the center
            x_ctrl = r_ctrl * cos(angm)
            y_ctrl = r_ctrl * sin(angm)
        else
            r_ctrl = 0.1 * row[edge_color_col]
            ang0 = π / 6
            ang1 = π / 6
            angm = (ang0 + ang1) / 2
            x_ctrl = xm + r_ctrl * cos(angm)
            y_ctrl = ym + r_ctrl * sin(angm)
        end
        ts = range(0, 1, length=n_curve_points)
        bx = (1 .- ts) .^ 2 .* x0 .+ 2 .* (1 .- ts) .* ts .* x_ctrl .+ ts .^ 2 .* x1
        by = (1 .- ts) .^ 2 .* y0 .+ 2 .* (1 .- ts) .* ts .* y_ctrl .+ ts .^ 2 .* y1
        c = (edge_color_scale[row.ec], edge_alpha)
        lines!(ax, bx, by, color=c, linewidth=edge_width)
    end

    if node_color_col isa Symbol
        @assert String(node_color_col) ∈ names(nodes_df)
        node_color_col = nodes_df[!, node_color_col]
    end

    # draw nodes after the edges
    sc = scatter!(
        ax,
        pos.x,
        pos.y,
        color=node_color_col,
        markersize=markersize,
        markerspace=:data,
        strokewidth=0.5,
        strokecolor=node_strokecolor,
    )

    # draw a circle is pos layout
    if layout == position_layout
        cm = (x=mean(pos.x), y=mean(pos.y))
        cm = (x=0.0, y=0.0)
        θ = LinRange(0, 2π, 36)
        x = cm.x .+ cos.(θ)
        y = cm.y .+ sin.(θ)
        lines!(ax, x, y, color=(:gray21, 0.2), linestyle=:dash
        )
    end


    # color bar
    # cmin, cmax = extrema(edges_df[!, :len])
    cb = Colorbar(fig[1, 2], limits=(ecmin, ecmax),
        height=200, width=15,
        colormap=edge_color_scale, vertical=true,
        label="Edge length",
        halign=:right,
        valign=:center, flip_vertical_label=true,
        scale=log10,
        tellheight=false, tellwidth=true,
    )


    # translate!(cb.blockscene, 450, 230, 1)
    hidespines!(ax)
    hidedecorations!(ax)

    fig, ax
end


end

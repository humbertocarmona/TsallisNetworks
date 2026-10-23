# `sortby` is a column of df, or a vector of columns
function check_sortby(df::DataFrame, sortby)
    for c in (sortby isa AbstractVector ? sortby : [sortby])
        @assert String(c) in names(df) "$c must be a column in df"
    end
end

"""
    circular(df::DataFrame; sortby=nothing, R=1.0) -> (x=..., y=...)

    Place the nodes (rows of `df`) on a circle of radius `R`. With `sortby`
    (a column or vector of columns), nodes are placed counterclockwise from
    angle 0 in decreasing order of those columns; positions are returned in
    the original row order.
"""
function circular(df::DataFrame; sortby=nothing, R=1.0)::NamedTuple
    isnothing(sortby) || check_sortby(df, sortby)

    n = nrow(df)
    layout = circular(n; R=R)

    # keep the original df intact
    dfc = copy(df)
    dfc[!, :id] = collect(1:n)

    if ~isnothing(sortby)
        sort!(dfc, sortby; rev=true)
    end

    dfc[!, :x] = layout.x
    dfc[!, :y] = layout.y

    # sort back
    dfc = @orderby(dfc, :id)

    return (x=dfc[:, :x], y=dfc[:, :y])
end


function circular(n::Int64; R=1.0)::NamedTuple
    θ = LinRange(0, 2π, n)
    x = R * cos.(θ)
    y = R * sin.(θ)

    return (x=x, y=y)
end

"""
    position_layout(df::DataFrame; sortby=nothing) -> (x=..., y=...)

    Use the node positions in the `:x` and `:y` columns of `df`. `sortby` is
    only checked: positions are fixed, and they must stay in row order, since
    edges refer to nodes by row index.
"""
function position_layout(df::DataFrame; sortby=nothing)::NamedTuple
    isnothing(sortby) || check_sortby(df, sortby)

    return (x=df[!, :x], y=df[!, :y])
end

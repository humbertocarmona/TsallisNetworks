function add_node!(G::Graphs.AbstractGraph,
    nodepositions::Vector{Vector{Float64}},
    edgeweights::SparseMatrixCSC{Float64},
    m::Int;
    γ=1.0,
    d=2,
    αA=2.0,
    αG=2.0,
    rng=Random.default_rng(),
    cm=true,
    rmin=1.0,
    rmax=1000.0)

    @assert αG >= 0 "αG must be non-negative."
    @assert αA >= 0 "αA must be non-negative."
    @assert d > 0 "dimension must be positive."

    # nodedegress is computed each time before adding new node
    nodedegrees = degree.(Ref(G), 1:nv(G))

    center = zeros(d)
    if cm
        center = mean(nodepositions)
    end
    pos_r = rand_radius(d, d + αG, rmin, rmax; rng=rng)
    new_pos = center + pos_r * rand_hypersphere(d; rng=rng)
    
    # before pushing new_pos!!!
    v_list = attach_list(new_pos, nodepositions, nodedegrees, αA, m; γ=γ, rng=rng)

    # now add the vertex to the graph
    add_vertex!(G)

    # add the new position to nodeposistions
    push!(nodepositions, new_pos)

    current_node = nv(G)
    #add the egdes
    for v in v_list
        e = add_edge!(G, current_node, v)
        if e
            edgeweights[current_node,v] = norm(nodepositions[current_node] - nodepositions[v])
            edgeweights[v,current_node] = edgeweights[current_node,v] 
        end
    end

    return nv(G), ne(G)
end
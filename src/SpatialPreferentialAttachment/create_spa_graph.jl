function create_spa_graph(N::Int64, m::Int64;
    γ::Float64=1.0,
    d::Int64=2,
    αA::Float64=2.0,
    αG::Float64=2.0,
    rmin::Float64=1.0,
    rmax::Float64=1000.0,
    cm::Bool=true,
    rng=Random.default_rng())

    @assert αG >= 0 "αG must be non-negative."
    @assert αA >= 0 "αA must be non-negative."
    @assert d > 0 "dimension must be positive."
    @assert γ <= 1

    g, seedpositions, seedweights = creategraph(N; d=d, m0=m, rmin=rmin, rng=rng)

    # growth state, kept incrementally instead of recomputed at each step
    P = zeros(d, max(N, nv(g)))                 # node positions as columns
    for (i, p) in enumerate(seedpositions)
        P[:, i] = p
    end
    degrees = zeros(Int, size(P, 2))
    degrees[1:nv(g)] = degree(g)
    I, J, V = findnz(seedweights)               # edge weights as triplets
    sizehint!(I, 2 * m * N)
    sizehint!(J, 2 * m * N)
    sizehint!(V, 2 * m * N)

    weights = zeros(size(P, 2))
    v_list = Vector{Int}(undef, m)
    new_pos = zeros(d)
    # sum of the positions of the existing nodes: a new node is placed at distance
    # r from their center of mass (cm = true, as in Brito et al., Sci. Rep. 6,
    # 27992 (2016), and add_node!), or from the origin (cm = false)
    possum = vec(sum(view(P, :, 1:nv(g)); dims=2))

    while nv(g) < N
        n = nv(g)
        # same draws, in the same order, as add_node!
        pos_r = rand_radius(d, d + αG, rmin, rmax; rng=rng)
        new_pos .= pos_r .* rand_hypersphere(d; rng=rng)
        cm && (new_pos .+= possum ./ n)

        attach_weights!(weights, P, new_pos, degrees, n, αA, γ)
        sample_norep!(v_list, rng, weights, n)

        add_vertex!(g)
        current_node = n + 1
        P[:, current_node] = new_pos
        possum .+= new_pos
        for v in v_list
            add_edge!(g, current_node, v)
            degrees[current_node] += 1
            degrees[v] += 1
            rij = norm(view(P, :, current_node) - view(P, :, v))
            push!(I, current_node, v)
            push!(J, v, current_node)
            push!(V, rij, rij)
        end

        if (nv(g) % 1000) == 0
            info(logger, "create_spa_graph:added node nn: $(nv(g)), ne: $(ne(g))")
        end
    end

    nodepositions = [P[:, i] for i in 1:nv(g)]
    edgeweights = sparse(I, J, V, N, N)
    return g, nodepositions, edgeweights
end

# ---------------------------------------------------------------------------
# Degree-degree correlations
# ---------------------------------------------------------------------------

"""
    degree_correlations(g)

Degree-degree correlations of `g`: `knn`, the named tuple (k, knn, n) of
`knn_table`; Pearson R over the edge ends (Newman's assortativity, equal to
`Graphs.assortativity`); Spearman ρ over the edge ends; and μ = ⟨k²⟩/⟨k⟩ over
the nodes of degree ≥ 1, the k_nn of an uncorrelated network.
Both orientations of every edge are used, so the two ends play the same role.
"""
function degree_correlations(g::SimpleGraph)
    deg = degree(g)
    M = ne(g)
    a = Vector{Int}(undef, 2M)
    b = Vector{Int}(undef, 2M)
    for (i, e) in enumerate(edges(g))
        ks, kd = deg[src(e)], deg[dst(e)]
        a[i], b[i] = ks, kd
        a[M+i], b[M+i] = kd, ks
    end
    k = filter(>(0), deg)
    return (knn=knn_table(g), R=cor(a, b), rho=corspearman(a, b),
        mu=mean(k .^ 2) / mean(k))
end

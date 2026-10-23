# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

"""
    clustering(g)

Average local clustering coefficient ⟨C⟩ (Brito et al., PRE 99, 012305
(2019), Eq. 5) and global transitivity T = 3 × triangles / connected triples.
Nodes of degree < 2 have local clustering 0 and count in the average.
Returns a named tuple (C, T).
"""
function clustering(g::SimpleGraph)
    return (C=mean(local_clustering_coefficient(g)), T=global_clustering_coefficient(g))
end

"""
    clustering_spectrum(g)

C(k), the mean local clustering of the nodes of degree k, as a named tuple of
vectors (k, Ck, n), with n the number of nodes of degree k, sorted by k.
"""
function clustering_spectrum(g::SimpleGraph)
    c = local_clustering_coefficient(g)
    acc = Dict{Int,Tuple{Float64,Int}}()
    for (k, ci) in zip(degree(g), c)
        s, n = get(acc, k, (0.0, 0))
        acc[k] = (s + ci, n + 1)
    end
    ks = sort!(collect(keys(acc)))
    return (k=ks, Ck=[acc[k][1] / acc[k][2] for k in ks], n=[acc[k][2] for k in ks])
end

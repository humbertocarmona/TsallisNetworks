using TsallisNetworks.Observables
using Graphs
using Random
using Statistics
using StatsBase
using Test

# the giant-component ⟨ℓ⟩ of the original ConfigurationModel implementation
function ell_giant_reference(g; nsources, rng)
    cc = connected_components(g)
    h, _ = induced_subgraph(g, cc[argmax(length.(cc))])
    n = nv(h)
    sources = n <= nsources ? collect(1:n) : sample(rng, 1:n, nsources; replace=false)
    return sum(sum(gdistances(h, s)) for s in sources) / (length(sources) * (n - 1))
end

@testset "observables" begin
    rng = Xoshiro(1)

    @testset "mean_shortest_path" begin
        @test mean_shortest_path(path_graph(5)) ≈ 2.0
        # two components, P5 and P3: mutually reachable pairs only (20 + 6 ordered
        # pairs, distance sums 40 + 8), or the largest component only
        g = blockdiag(path_graph(5), path_graph(3))
        @test mean_shortest_path(g) ≈ 48 / 26
        @test mean_shortest_path(g; giant=true) ≈ 2.0
        # sampled sources: same draws and value as the original implementation
        g = erdos_renyi(2000, 2400; rng=rng)
        @test !is_connected(g)
        for threaded in (true, false)
            @test mean_shortest_path(g; nsources=300, giant=true, threaded, rng=Xoshiro(5)) ≈
                  ell_giant_reference(g; nsources=300, rng=Xoshiro(5))
        end
        @test mean_shortest_path(g; threaded=true) ≈ mean_shortest_path(g; threaded=false)
    end

    @testset "clustering" begin
        @test clustering(complete_graph(5)) == (C=1.0, T=1.0)
        @test clustering(star_graph(6)) == (C=0.0, T=0.0)
        g = barabasi_albert(3000, 3; rng=rng)
        c = clustering(g)
        @test c.C ≈ mean(local_clustering_coefficient(g))
        @test c.T ≈ global_clustering_coefficient(g)
        ck = clustering_spectrum(g)
        @test issorted(ck.k) && sum(ck.n) == nv(g)
        @test sum(ck.Ck .* ck.n) / nv(g) ≈ c.C
        # triangle with a pendant node: degrees 2, 2, 3, 1
        g = SimpleGraph(Edge.([(1, 2), (2, 3), (1, 3), (3, 4)]))
        @test clustering_spectrum(g) == (k=[1, 2, 3], Ck=[0.0, 1.0, 1 / 3], n=[1, 2, 1])
    end

    @testset "degree_correlations" begin
        # star: hub of degree 5 linked only to leaves, perfectly disassortative
        dc = degree_correlations(star_graph(6))
        @test dc.R ≈ -1 && dc.rho ≈ -1
        @test dc.mu ≈ (25 + 5) / 10
        @test dc.knn == (k=[1, 5], knn=[5.0, 1.0], n=[5, 1])
        g = barabasi_albert(3000, 2; rng=rng)
        dc = degree_correlations(g)
        @test dc.R ≈ assortativity(g)
        @test (dc.knn.k, dc.knn.knn) == knn(g)
        @test sum(dc.knn.n) == nv(g)
    end

    @testset "robustness" begin
        N = 50
        # complete graph: S = (N - Q)/N after Q removals, in any order
        r = robustness(complete_graph(N); rng=rng)
        @test r.S ≈ (N:-1:0) ./ N
        @test r.f == (0:N) ./ N
        @test r.Rindex ≈ (N - 1) / (2N)
        # star, targeted: the hub goes first and the network falls apart at once
        r = robustness(star_graph(N); attack=:targeted, rng=rng)
        @test r.S[2] == 1 / N
        @test r.fc_jump == 1 / N && r.fc_mr == 1 / N
        @test robustness(star_graph(N); attack="targeted", rng=Xoshiro(2)).S == r.S
        # same draws as percolation with randperm / attack_order
        g = barabasi_albert(1000, 2; rng=rng)
        for (attack, order) in ((:random, randperm(Xoshiro(3), nv(g))),
            (:targeted, attack_order(g; rng=Xoshiro(3))))
            r = robustness(g; attack, rng=Xoshiro(3))
            S, κ = percolation(g, order)
            @test r.S == S && r.κ == κ
            @test r.fc_mr == molloy_reed_fc(κ)
        end
        # hubs first breaks the network sooner than random failure
        @test robustness(g; attack=:targeted, rng=rng).Rindex < robustness(g; rng=rng).Rindex
        @test_throws ArgumentError robustness(g; attack=:degree)
    end
end

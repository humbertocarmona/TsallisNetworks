using TsallisNetworks.ConfigurationModel
using Random
using Graphs
using Statistics
using StatsBase
using Test

@testset "null model" begin
    rng = Xoshiro(1)

    @testset "q-exponential degrees" begin
        q, λ = 1.4, 1.0
        degs = sample_qexp_degrees(200_000, q, λ; rng=rng)
        @test minimum(degs) >= 2
        @test iseven(sum(degs))
        @test sum(qexp_pk(k, q, λ) for k in 2:(10^7)) ≈ 1 atol = 1e-3
        for k in 2:6
            @test mean(degs .== k) ≈ qexp_pk(k, q, λ) rtol = 0.03
        end
    end

    @testset "stub matching" begin
        degs = sample_qexp_degrees(20_000, 1.3, 1.0; rng=rng)
        g, left = stub_matching(degs; rng=rng)
        @test !has_self_loops(g)
        @test ne(g) == (sum(degs) - left) ÷ 2
        @test all(degree(g) .<= degs)
        # pathological sequence: all stubs on one node must not hang
        g2, left2 = stub_matching([4, 0, 0]; rng=rng)
        @test ne(g2) == 0 && left2 == 4
    end

    @testset "rewiring" begin
        g = barabasi_albert(5000, 2; rng=rng)
        h, acc = rewire(g; rng=rng)
        @test degree(h) == degree(g)
        @test ne(h) == ne(g)
        @test !has_self_loops(h)
        @test acc > 0.5
        @test length(intersect(Set(edges(h)), Set(edges(g)))) < 0.05 * ne(g)
    end

    @testset "observables" begin
        # a grown graph with m links per node is m-degenerate
        g = barabasi_albert(5000, 2, 2; complete=true, rng=rng)
        kh, Mh = kcore_stats(g)
        @test kh == 2 && Mh == nv(g)

        # percolation against connected_components on a small graph
        g = erdos_renyi(300, 600; rng=rng)
        order = randperm(rng, nv(g))
        S, κ = percolation(g, order)
        for nrem in (0, 50, 150, 299)
            keep = order[(nrem+1):end]
            h, _ = induced_subgraph(g, keep)
            cc = connected_components(h)
            @test S[nrem+1] ≈ maximum(length.(cc)) / nv(g)
        end
        @test S[end] == 0

        # adaptive attack always removes a node of maximum current degree
        g = barabasi_albert(500, 2; rng=rng)
        order = attack_order(g; rng=rng)
        @test sort(order) == 1:nv(g)
        h = copy(g)
        alive = trues(nv(g))
        for v in order[1:100]
            @test degree(h, v) == maximum(degree(h, w) for w in vertices(h) if alive[w])
            for u in collect(neighbors(h, v))
                rem_edge!(h, u, v)
            end
            alive[v] = false
        end

        # exact ⟨ℓ⟩ when all nodes are sources
        g = path_graph(5)
        @test mean_shortest_path(g; nsources=10) ≈ 2.0

        ks, kn = knn(star_graph(5))
        @test ks == [1, 4] && kn == [4.0, 1.0]
    end
end

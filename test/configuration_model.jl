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

    @testset "degree sampler boundaries and stable probabilities" begin
        for N in (0, -1)
            @test_throws ArgumentError sample_qexp_degrees(N, 1.3, 1.0)
            @test_throws ArgumentError sample_qexp_degrees_discrete(N, 1.3, 1.0)
        end
        @test_throws ArgumentError sample_qexp_degrees(3, 1.3, 1.0; kmin=3)
        @test_throws ArgumentError sample_qexp_degrees_discrete(3, 1.3, 1.0; kmin=1, kmax=1)
        for scale in (0.0, -1.0, Inf, NaN)
            @test_throws ArgumentError sample_qexp_degrees(10, 1.3, scale)
            @test_throws ArgumentError sample_qexp_degrees_discrete(10, 1.3, scale)
        end
        @test sample_qexp_degrees(1, 1.0, 1.0; kmin=0) == [0]
        @test sample_qexp_degrees_discrete(4, 1.3, 1.0; kmin=1, kmax=1) == ones(Int, 4)

        # Previously every raw weight underflowed, silently producing only 800.
        d = sample_qexp_degrees_discrete(100_000, 1.0, 1.0;
            kmin=800, kmax=805, rng=Xoshiro(42))
        z = sum(exp(-j) for j in 0:5)
        for k in 800:805
            p = exp(-(k - 800)) / z
            @test abs(mean(d .== k) - p) < 5sqrt(p * (1 - p) / length(d)) + 1 / length(d)
        end
        @test iseven(sum(d))
        # Conditional inverse sampling must not stall in an extremely rare tail.
        d = sample_qexp_degrees(1000, 1.0, 1.0; kmin=800, rng=Xoshiro(43))
        @test all(800 .<= d .<= 999) && iseven(sum(d))
        @test abs(mean(d .== 800) - (1 - exp(-1))) < 0.07

        # With effectively one numerical parity, repair still terminates.
        d = sample_qexp_degrees_discrete(3, 1.0, 0.001; kmin=1, kmax=2, rng=Xoshiro(44))
        @test sum(d) == 4
        for q in (1.0, 1.00000001, 1.5)
            @test sum(qexp_pk(k, q, 1.0; kmin=2, kmax=9) for k in 2:9) ≈ 1
            @test qexp_pk(1, q, 1.0; kmin=2, kmax=9) == 0
            @test qexp_pk(10, q, 1.0; kmin=2, kmax=9) == 0
        end
        @test qexp_pk(9, 1.5, 1.0; kmin=2, kmax=9) ≈ 4 / 11
        @test qexp_pk(800, 1.0, 1.0; kmin=800) ≈ 1 - exp(-1)
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
        for g0 in (SimpleGraph(0), SimpleGraph(3), path_graph(3))
            h0, acc0 = rewire(g0; swaps_per_edge=0)
            @test h0 == g0 && h0 !== g0 && acc0 == 0.0
        end
        @test rewire(SimpleGraph(3))[2] == 0.0
        @test_throws ArgumentError rewire(g; swaps_per_edge=-1)
        @test_throws ArgumentError rewire(g; swaps_per_edge=Inf)
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

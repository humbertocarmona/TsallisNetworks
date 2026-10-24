using TsallisNetworks.EdgeRecruitment
using Graphs
using Random
using Statistics
using Test

@testset "edge recruitment" begin
    @testset "physical multigraph and projection" begin
        for L in (0, 1, 3), n0 in (2, 3, 5)
            N = 500
            g = edge_recruitment_graph(N; L, n0, rng=Xoshiro(101))
            @test nv(g) == N
            @test ne(g) == n0 * (n0 - 1) ÷ 2 + 2 * (N - n0)
            @test sum(degree(g)) == 2ne(g)
            dist = recruitment_degree_distribution(g)
            @test sum(dist.n) == N && sum(dist.pk) ≈ 1
            @test sum(dist.k .* dist.n) == 2ne(g)
            @test all(dist.n[k+1] == count(==(k), degree(g)) for k in dist.k)
            @test all(src(e) != dst(e) for e in edges(g))
            @test length(collect(edges(g))) == ne(g)
            @test all(sum(u < t for u in g.adjacency[t]) == 2 for t in (n0+1):N)
            @test all(degree(g, v) == length(g.adjacency[v]) for v in vertices(g))
            h = simple_projection(g)
            @test nv(h) == N && is_connected(h)
            @test all(degree(h) .<= degree(g))
            if L == 0
                @test ne(h) == ne(g)
                @test degree(h) == degree(g)
            else
                @test ne(h) < ne(g)
            end
        end
        g = edge_recruitment_graph(300; L=2, rng=Xoshiro(9), T=Int32)
        h = edge_recruitment_graph(300; L=2, rng=Xoshiro(9))
        @test g.edge_list == h.edge_list && g.adjacency == h.adjacency
        @test eltype(first(g.adjacency)) == Int32
        @test g.edge_list == edge_recruitment_graph(300; L=2, rng=Xoshiro(9), T=Int32).edge_list
        seed = edge_recruitment_graph(3; L=7, n0=3)
        @test ne(seed) == 3 && degree(seed) == [2, 2, 2]
        @test edge_multiplicity(seed, 1, 2) == 1
        @test edge_multiplicity(seed, 1, 1) == 0
        @test_throws BoundsError edge_multiplicity(seed, 1, 4)
    end

    @testset "sampling opportunities and endpoint increments" begin
        # K3 has 3 physical edges and 6 virtual loops when L=2.
        # Probability of a double edge at the first arrival is 2/3.
        # Each existing vertex has expected increment (2+4)/9 = 2/3.
        rng = Xoshiro(23)
        trials = 12_000
        doubles = 0
        increments = zeros(3)
        multiplicities_valid = true
        for _ in 1:trials
            g = edge_recruitment_graph(4; L=2, n0=3, rng)
            doubles += g.adjacency[4][1] == g.adjacency[4][2]
            increments .+= degree(g)[1:3] .- 2
            for v in unique(g.adjacency[4])
                multiplicities_valid &= edge_multiplicity(g, 4, v) == count(==(v), g.adjacency[4])
            end
        end
        @test multiplicities_valid
        @test abs(doubles / trials - 2 / 3) < 0.025
        @test all(abs.(increments ./ trials .- 2 / 3) .< 0.025)
    end

    @testset "stationary discrete law and continuum parameters" begin
        ks = 2:100
        @test recruitment_pk(0, 100)[ks] ≈ 12 ./ (ks .* (ks .+ 1) .* (ks .+ 2))
        @test isempty(recruitment_pk(1, 0))
        @test recruitment_pk(1, 1) == [0.0]
        for L in (0, 1, 3)
            p = recruitment_pk(L, 100_000)
            @test sum(p) ≈ 1 atol=1e-8
            @test sum(k * p[k] for k in eachindex(p)) ≈ 4 atol=0.001
            @test p[2] ≈ (L + 2) / (2L + 4)
        end
        @test recruitment_qexp(1) == (q=1.25, κ=0.5)
        @test recruitment_qexp(0) == (q=4/3, κ=0.0)
        # Independent ensembles test the birth/jump law beyond a seed step.
        for L in (0, 1, 3)
            samples = [degree(edge_recruitment_graph(10_000; L, rng=Xoshiro(s))) for s in 1:8]
            p = recruitment_pk(L, 10)
            for k in 2:8
                freq = mean(count(==(k), d) / length(d) for d in samples)
                @test abs(freq - p[k]) < 0.015
            end
        end
    end

    @testset "invalid arguments" begin
        @test_throws ArgumentError edge_recruitment_graph(10; L=-1)
        @test_throws ArgumentError edge_recruitment_graph(10; n0=1)
        @test_throws ArgumentError edge_recruitment_graph(2; n0=3)
        @test_throws ArgumentError edge_recruitment_graph(200; T=Int8)
        @test_throws ArgumentError recruitment_pk(-1, 10)
        @test_throws ArgumentError recruitment_pk(1, -1)
        @test_throws ArgumentError recruitment_qexp(-1)
    end
end

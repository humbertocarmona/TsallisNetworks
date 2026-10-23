using TsallisNetworks: expq
using TsallisNetworks.PreferentialRandom
using Graphs
using Random
using Test

@testset "preferential random model" begin
    @testset "structure" begin
        for (α, n0) in ((0.0, 3), (0.5, 2), (1.0, 5))
            N = 5000
            g = preferential_random_graph(N, α; n0, rng=Xoshiro(1))
            @test nv(g) == N
            @test ne(g) == n0 * (n0 - 1) ÷ 2 + (N - n0)    # one edge per new node
            @test is_connected(g)
            @test !has_self_loops(g)
            @test all(issorted, g.fadjlist)
            # each new node links to exactly one older node
            @test all(count(<(t), neighbors(g, t)) == 1 for t in (n0+1):N)
        end
        # n0 = 3: total degree M(t) = 2t, as in the rate equation
        @test sum(degree(preferential_random_graph(1000, 0.4; rng=Xoshiro(2)))) == 2000
        g = preferential_random_graph(1000, 0.5; rng=Xoshiro(3), T=Int32)
        @test eltype(g) == Int32
        @test g == preferential_random_graph(1000, 0.5; rng=Xoshiro(3))
        @test_throws AssertionError preferential_random_graph(100, 1.5)
    end

    @testset "rate equation: limiting cases" begin
        k = 1:50
        @test sum(pr_pk(0.6, 10^6)) ≈ 1 atol = 1e-6
        # α = 0: random recursive tree, n_k = 2^-k
        @test pr_pk(0.0, 50) ≈ 0.5 .^ k
        # α = 1: Barabási-Albert with m = 1, n_k = 4/(k(k+1)(k+2))
        @test pr_pk(1.0, 50) ≈ 4 ./ (k .* (k .+ 1) .* (k .+ 2))
    end

    @testset "q-exponential: large-k limit of the rate equation" begin
        # n_k / e_q(-k/κ) tends to a constant, with q of Eq. (21)
        for α in (0.2, 0.5, 0.9)
            (; q, κ) = pr_qexp(α)
            n = pr_pk(α, 10^5)
            r = [n[k] / expq(-k / κ, q) for k in (10^3, 10^4, 10^5)]
            @test r[2] ≈ r[1] rtol = 1e-4
            @test r[3] ≈ r[1] rtol = 1e-4
            @test 1 / (q - 1) ≈ 1 + 2 / α                   # tail exponent
        end
    end

    # |observed frequency - p| within `z` binomial standard deviations
    within(count, n, p; z=5) = abs(count / n - p) <= z * sqrt(p * (1 - p) / n)

    @testset "simulation: degree distribution (α = $α)" for α in (0.0, 0.5, 0.9, 1.0)
        degs = reduce(vcat, [degree(preferential_random_graph(500_000, α; rng=Xoshiro(10 + s)))
                             for s in 1:4])
        n = length(degs)
        p = pr_pk(α, 200)
        ks = filter(k -> n * p[k] >= 20, 1:200)              # well populated degrees
        @test length(ks) >= 8
        @test all(within(count(==(k), degs), n, p[k]) for k in ks)
    end

    @testset "simulation: q-exponential tail (α = $α)" for α in (0.5, 0.9)
        # MLE of γ for P(k) ∝ (k + κ/(q-1))^-γ = e_q(-k/κ) on k ≥ kmin, in the
        # continuous approximation; γ = 1/(q - 1) gives q of Eq. (21)
        (; q, κ) = pr_qexp(α)
        c, kmin = κ / (q - 1), 20
        tail = Float64[]
        for s in 1:4
            k = degree(preferential_random_graph(500_000, α; rng=Xoshiro(20 + s)))
            append!(tail, filter(>=(kmin), k))
        end
        γ = 1 + length(tail) / sum(log.((tail .+ c) ./ (kmin - 0.5 + c)))
        σ = (γ - 1) / sqrt(length(tail))
        @test abs(γ - 1 / (q - 1)) <= 5σ
        @test abs((1 + 1 / γ) - q) <= 5σ / γ^2              # q̂ = 1 + 1/γ̂
    end
end

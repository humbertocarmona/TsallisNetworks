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

    @testset "multi-edge extension" begin
        for m0 in (2, 4), α in (0.0, 0.5, 1.0)
            N, n0 = 2000, m0 + 1
            g = preferential_random_graph(N, α; m0, rng=Xoshiro(31), T=Int32)
            @test nv(g) == N
            @test ne(g) == n0*(n0-1)÷2 + m0*(N-n0)
            @test all(count(<(t), neighbors(g,t)) == m0 for t in (n0+1):N)
            @test is_connected(g) && !has_self_loops(g)
            @test all(issorted, g.fadjlist)
            @test minimum(degree(g)) >= m0
            @test g == preferential_random_graph(N, α; m0, rng=Xoshiro(31))
            p = pr_pk(α, 100_000; m0)
            @test all(iszero, p[1:m0-1])
            @test sum(p) ≈ 1 atol=1e-7
            @test sum(k*p[k] for k in eachindex(p)) ≈ 2m0 atol=0.001
            if α == 0
                @test p[m0:50] ≈ [1/(1+m0)*(m0/(1+m0))^(k-m0) for k in m0:50]
            elseif α == 1
                @test p[m0:50] ≈ [2m0*(m0+1)/(k*(k+1)*(k+2)) for k in m0:50]
            end
        end
        for m0 in (1, 2, 4)
            @test pr_qexp(0.6; m0).κ ≈ (2*0.4*m0+1)/2.6
            @test pr_qexp(0.6; m0, approximation=:continuum).κ ≈ 2*0.4*m0/2.6
            @test pr_qexp(0.6; m0).q ≈ pr_qexp(0.6).q
            @test pr_qexp(1.0; m0, approximation=:continuum).κ == 0
            @test pr_qexp(0.0; m0).κ ≈ 1/log1p(1/m0)
        end
        @test isempty(pr_pk(0.5, 0; m0=2))
        @test pr_pk(0.5, 1; m0=2) == [0.0]
        @test_throws AssertionError preferential_random_graph(100, 0.5; m0=0)
        @test_throws AssertionError preferential_random_graph(100, 0.5; m0=4, n0=3)
        @test_throws AssertionError pr_pk(0.5, 10; m0=0)
        @test_throws ArgumentError pr_qexp(0.5; approximation=:unknown)

        # After the first m0=2 birth on K3, old degrees are {3,3,2,2}.
        # Check the probability of choosing both high-degree vertices in the
        # next birth against sequential mixed-weight sampling without replacement.
        rng, trials, highpairs = Xoshiro(32), 8000, 0
        α = 0.6
        for _ in 1:trials
            g = preferential_random_graph(5, α; m0=2, n0=3, rng)
            targets = neighbors(g, 5)
            highpairs += all(degree(g,v)-1 == 3 for v in targets)
        end
        ph = α*3/10 + (1-α)/4
        expected = 2ph^2/(1-ph)
        @test abs(highpairs/trials-expected) <= 5sqrt(expected*(1-expected)/trials)

        # Limiting recurrence versus independent finite growing ensembles.
        for α in (0.0, 0.5, 1.0), m0 in (2, 4)
            degs = reduce(vcat, [degree(preferential_random_graph(50_000, α;
                m0, rng=Xoshiro(100+s))) for s in 1:4])
            p = pr_pk(α, 30; m0)
            @test all(abs(count(==(k),degs)/length(degs)-p[k]) < 0.008 for k in m0:20)
        end
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

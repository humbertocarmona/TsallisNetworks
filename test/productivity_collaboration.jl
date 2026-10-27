using TsallisNetworks.ProductivityCollaboration
using Random, Statistics, Test

@testset "productivity-driven collaboration" begin
    @testset "conditional geometric law" begin
        @test collaboration_mean(1; b=0.33, beta=1.3) == 1
        @test collaboration_mean(10; b=2, beta=1) ≈ 20
        @test collaboration_pmf(0, 10) == 0
        @test collaboration_pmf(1, 1) == 1
        @test collaboration_pmf(2, 1) == 0
        @test sample_collaborators(1) == 1
        p = [collaboration_pmf(k, 10; b=2) for k in 1:5000]
        @test sum(p) ≈ 1
        @test sum(k*p[k] for k in eachindex(p)) ≈ 20
        @test sum((k-20)^2*p[k] for k in eachindex(p)) ≈ 20*19
        samples = productivity_collaboration(fill(10, 100_000); b=2, rng=Xoshiro(70))
        d = samples.collaborators
        @test abs(mean(d)-20) < 0.4
        @test abs(var(d)-380) < 20
        for k in 1:10
            prob = p[k]
            @test abs(mean(d .== k)-prob) < 5sqrt(prob*(1-prob)/length(d))
        end
        input = [1.0, 2.0, 10.0]
        r = productivity_collaboration(input; rng=Xoshiro(8))
        @test r.productivity == input && r.productivity !== input
        @test r == productivity_collaboration(input; rng=Xoshiro(8))
        @test isempty(productivity_collaboration(0).collaborators)
    end

    @testset "lognormal productivity and integrated mixture" begin
        r = productivity_collaboration(100_000; logmean=1.0, logsigma=0.5, rng=Xoshiro(71))
        @test abs(mean(log.(r.productivity))-1) < 0.01
        @test abs(std(log.(r.productivity))-0.5) < 0.01
        @test all(r.collaborators .>= 1)
        ks = collect(1:15)
        p = collaboration_mixture_pmf(ks; logmean=1, logsigma=0.5)
        for k in ks
            @test abs(mean(r.collaborators .== k)-p[k]) < 5sqrt(p[k]*(1-p[k])/length(r.collaborators))
        end
        @test collaboration_mixture_pmf(ks; logmean=1, logsigma=0.5, intervals=2048) ≈ p rtol=1e-6
        @test sum(collaboration_mixture_pmf(collect(1:1000); logmean=1, logsigma=0.5)) ≈ 1 atol=1e-7
        @test collaboration_mixture_pmf(ks; logmean=log(4), logsigma=0) ≈
            [collaboration_pmf(k, 4) for k in ks]
        @test collaboration_mixture_pmf([0, 1, 2]; beta=0, b=1) == [0.0, 1.0, 0.0]
        # Low-productivity plateau carries genuine probability at C=1.
        p1 = collaboration_mixture_pmf([1]; logmean=-1, logsigma=0.5)[1]
        @test 0.97 < p1 <= 1
    end

    @testset "exact exposure-time paths" begin
        A = [0.1, 0.5, 1.0, 2.0, 2.0, 5.0, 10.0]
        path = collaboration_trajectory(A; rng=Xoshiro(72))
        @test issorted(path.collaborators)
        @test path.collaborators[1:3] == [1, 1, 1]
        @test path.collaborators[4] == path.collaborators[5]
        @test path.exposure ≈ log.(path.conditional_mean)
        @test path == collaboration_trajectory(A; rng=Xoshiro(72))
        @test last(path.collaborators) == only(collaboration_trajectory([10.0]; rng=Xoshiro(72)).collaborators)
        @test isempty(collaboration_trajectory(Float64[]).collaborators)
        rng = Xoshiro(73)
        endpoints = [only(collaboration_trajectory([4.0]; rng).collaborators) for _ in 1:20_000]
        @test abs(mean(endpoints)-4) < 0.1
        @test abs(var(endpoints)-12) < 0.6
        @test abs(mean(endpoints .== 1)-0.25) < 0.02
    end

    @testset "validation" begin
        for a in (0, -1, Inf, NaN)
            @test_throws ArgumentError sample_collaborators(a)
        end
        @test_throws ArgumentError collaboration_mean(10; b=0)
        @test_throws ArgumentError collaboration_mean(10; beta=-1)
        @test_throws ArgumentError productivity_collaboration(-1)
        @test_throws ArgumentError productivity_collaboration(10; logsigma=-1)
        @test_throws ArgumentError productivity_collaboration(10; logmean=Inf)
        @test_throws ArgumentError collaboration_trajectory([2, 1])
        @test_throws ArgumentError collaboration_mixture_pmf([1]; intervals=3)
        @test_throws ArgumentError collaboration_mixture_pmf([1]; zmax=0)
    end
end

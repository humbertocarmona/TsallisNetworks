using TsallisNetworks.SpatialPreferentialAttachment
using DataFrames
using Graphs
using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Test

import TsallisNetworks.SpatialPreferentialAttachment as SPA

# keep test output clean (create_spa_graph logs every 1000 nodes)
SPA.Memento.setlevel!(SPA.logger, "warn")

# |observed frequency - p| within `z` binomial standard deviations
within_binomial(count, n, p; z=5) = abs(count / n - p) <= z * sqrt(p * (1 - p) / n)

@testset "SpatialPreferentialAttachment.jl" begin

    @testset "rand_hypersphere" begin
        rng = MersenneTwister(1)
        for d in 1:4
            for _ in 1:100
                v = SPA.rand_hypersphere(d; rng=rng)
                @test length(v) == d
                @test norm(v) ≈ 1.0
            end
        end
        # directions are isotropic: mean vector ≈ 0
        vs = [SPA.rand_hypersphere(3; rng=rng) for _ in 1:20_000]
        @test all(abs.(mean(vs)) .< 0.03)
    end

    @testset "rand_radius" begin
        rng = MersenneTwister(2)
        rmin, rmax = 1.0, 1000.0
        n = 20_000
        # (d, α) with α == d (log-uniform) and α > d (power law)
        for (d, α) in ((2, 2.0), (2, 4.0), (3, 5.0))
            rs = [SPA.rand_radius(d, α, rmin, rmax; rng=rng) for _ in 1:n]
            @test all(rmin .<= rs .<= rmax)
            # exact CDF of the truncated power law
            β = d - α
            F = α == d ? (r -> log(r / rmin) / log(rmax / rmin)) :
                (r -> (r^β - rmin^β) / (rmax^β - rmin^β))
            for r in (1.5, 3.0, 10.0, 100.0)
                @test within_binomial(count(<=(r), rs), n, F(r))
            end
        end
        # degenerate interval returns rmin exactly
        @test SPA.rand_radius(2, 2.0, 3.0, 3.0; rng=rng) ≈ 3.0
        @test SPA.rand_radius(2, 4.0, 3.0, 3.0; rng=rng) ≈ 3.0
        @test_throws AssertionError SPA.rand_radius(3, 2.0, rmin, rmax; rng=rng)
    end

    @testset "expq / lnq" begin
        for x in (-0.5, 0.0, 0.3, 2.0)
            @test expq(x, 1.0) ≈ exp(x)
            @test lnq(exp(x), 1.0) ≈ x
            for q in (0.5, 1.5)
                y = expq(x, q)
                y > 0 && @test lnq(y, q) ≈ x
            end
        end
        @test expq(-3.0, 0.5) == 0.0          # cut-off: 1 + (1-q)x <= 0
        @test_throws AssertionError lnq(0.0, 1.5)
    end

    @testset "map_to_interval" begin
        x = [0.0, 5.0, 10.0]
        @test map_to_interval(x, 0.0, 10.0) ≈ [0.0, 0.5, 1.0]
        @test map_to_interval(x, 0.0, 10.0; a=-1.0, b=1.0) ≈ [-1.0, 0.0, 1.0]
        @test map_to_interval([-5.0, 15.0], 0.0, 10.0) ≈ [0.0, 1.0]   # clamped
    end

    @testset "layouts" begin
        df = DataFrame(k=[3, 9, 1, 5], c=[1, 1, 2, 2], x=[0.1, 0.2, 0.3, 0.4], y=[1.0, 2.0, 3.0, 4.0])
        ref = circular(4)
        # decreasing k: node 2 takes the first slot, then 4, 1, 3
        pos = circular(df; sortby=:k)
        @test pos.x == ref.x[[3, 1, 4, 2]] && pos.y == ref.y[[3, 1, 4, 2]]
        @test circular(df; sortby="k") == pos
        # several columns: decreasing c, then decreasing k
        pos = circular(df; sortby=[:c, :k])
        @test pos.x == ref.x[[4, 3, 2, 1]]
        @test circular(df) == (x=ref.x, y=ref.y)
        @test_throws AssertionError circular(df; sortby=:missing)
        # positions stay in row order whatever sortby is
        @test position_layout(df; sortby=:k) == (x=df.x, y=df.y)
        @test_throws AssertionError position_layout(df; sortby=[:k, :missing])
    end

    @testset "degree_distribution" begin
        degs = [1, 1, 2, 2, 3, 4, 8, 16]
        x, y = degree_distribution(degs)
        @test x == [1.0, 2.0, 3.0, 4.0, 8.0, 16.0]
        w = push!(diff(x), diff(x)[end])
        @test sum(y .* w) ≈ 1.0

        x, y = degree_distribution(degs; log_binning=true, nbins=5)
        bins = 2 .^ range(0, 4; length=5)
        @test length(x) == length(y) == 4
        @test sum(y .* diff(bins)) ≈ 1.0

        g = path_graph(5)
        @test degree_distribution(g) == degree_distribution(degree(g))
    end

    @testset "creategraph" begin
        N, d, m0, rmin = 50, 3, 4, 2.0
        g, pos, w = creategraph(N; d=d, m0=m0, rmin=rmin, rng=MersenneTwister(3))
        @test nv(g) == m0
        @test ne(g) == m0 * (m0 - 1) ÷ 2               # complete graph
        @test length(pos) == m0
        @test all(length(p) == d for p in pos)
        # on the sphere of radius rmin, up to the jitter ε = 1e-3 per coordinate
        @test all(abs(norm(p) - rmin) <= 1e-3 * sqrt(d) for p in pos)
        @test all(abs(norm(p) - rmin) <= 1e-12 for p in
                  creategraph(N; d=d, m0=m0, rmin=rmin, ε=0.0, rng=MersenneTwister(3))[2])
        @test size(w) == (N, N)
        @test w == w'
        @test nnz(w) == 2 * ne(g)
        for e in edges(g)
            @test w[e.src, e.dst] ≈ norm(pos[e.src] - pos[e.dst])
        end

        # m0 = 1 still starts from two connected nodes
        g, pos, w = creategraph(N; d=2, m0=1, rmin=rmin, rng=MersenneTwister(3))
        @test nv(g) == 2 && ne(g) == 1 && length(pos) == 2
        @test w[1, 2] == w[2, 1] ≈ norm(pos[1] - pos[2])
    end

    @testset "attach_list" begin
        rng = MersenneTwister(4)
        pos = [[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -1.0], [4.0, 4.0]]
        degs = [1, 5, 2, 3, 7]
        newpos = [0.0, 0.0]
        n = length(pos)

        # m distinct, valid indices
        for m in 1:n
            v = SPA.attach_list(newpos, pos, degs, 2.0, m; rng=rng)
            @test length(v) == m
            @test allunique(v)
            @test all(1 .<= v .<= n)
        end

        # a node with overwhelming weight is always chosen
        near = [[1e-6, 0.0], [5.0, 5.0], [-5.0, 5.0]]
        for _ in 1:50
            @test 1 in SPA.attach_list(newpos, near, [1, 1, 1], 2.0, 1; rng=rng)
        end

        # law of a single draw: Π_i ∝ k_i^γ / r_i^α
        ntrials = 40_000
        for (α, γ) in ((2.0, 1.0), (0.0, 1.0), (3.5, 0.5))
            p = (degs .^ γ) ./ (norm.(pos) .^ α)
            p ./= sum(p)
            counts = zeros(Int, n)
            for _ in 1:ntrials
                counts[only(SPA.attach_list(newpos, pos, degs, α, 1; γ=γ, rng=rng))] += 1
            end
            @test all(within_binomial(counts[i], ntrials, p[i]) for i in 1:n)
        end

        # law of m = 2: sequential draws without replacement,
        # P({i,j}) = p_i p_j / (1 - p_i) + p_j p_i / (1 - p_j)
        p = degs ./ (norm.(pos) .^ 2)
        p ./= sum(p)
        pairs = Dict{Tuple{Int,Int},Int}()
        for _ in 1:ntrials
            v = sort(SPA.attach_list(newpos, pos, degs, 2.0, 2; rng=rng))
            pairs[(v[1], v[2])] = get(pairs, (v[1], v[2]), 0) + 1
        end
        for i in 1:n-1, j in i+1:n
            pij = p[i] * p[j] / (1 - p[i]) + p[j] * p[i] / (1 - p[j])
            @test within_binomial(get(pairs, (i, j), 0), ntrials, pij)
        end

        @test_throws AssertionError SPA.attach_list(newpos, pos, degs, -1.0, 1; rng=rng)
        @test_throws AssertionError SPA.attach_list(newpos, pos, degs, 2.0, 1; γ=1.5, rng=rng)
        @test_throws AssertionError SPA.attach_list([0.0], pos, degs, 2.0, 1; rng=rng)
    end

    @testset "attach_weights! kernels" begin
        rng = MersenneTwister(10)
        # covers every kernel: α = 2, α = 0, half-integer α (sqrt path),
        # 2α > 16 and non-half-integer α (exp/log path), γ ≠ 1, d > 4
        for d in (1, 2, 3, 5), α in (0.0, 0.7, 1.5, 2.0, 3.0, 9.0), γ in (1.0, 0.5)
            n = 200
            P = randn(rng, d, n + 10) .* 10
            x = randn(rng, d)
            k = rand(rng, 1:30, n + 10)
            w = fill(-1.0, n + 10)
            SPA.attach_weights!(w, P, x, k, n, α, γ)
            exact = [k[i]^γ / norm(P[:, i] - x)^α for i in 1:n]
            @test w[1:n] ≈ exact rtol = 1e-12
            @test all(==(-1.0), w[n+1:end])             # only 1:n is written
        end
    end

    @testset "sample_norep!" begin
        rng = MersenneTwister(11)
        # one weight dominating by 1e30 must not break the following draws
        for _ in 1:100
            w = [1e30, 1.0, 2.0, 0.0, 3.0]
            v = SPA.sample_norep!(zeros(Int, 3), rng, w, 4)
            @test v[1] == 1 && sort(v[2:3]) == [2, 3]     # index 5 is outside 1:n
        end
        @test_throws ArgumentError SPA.sample_norep!(zeros(Int, 2), rng, [1.0, 0.0, 0.0], 3)
        @test_throws ArgumentError SPA.sample_norep!(zeros(Int, 1), rng, [Inf, 1.0], 2)
    end

    @testset "add_node!" begin
        N, m, d = 30, 3, 2
        g, pos, w = creategraph(N; d=d, m0=m, rng=MersenneTwister(5))
        rng = MersenneTwister(6)
        for step in 1:10
            n0, e0 = nv(g), ne(g)
            nn, nl = add_node!(g, pos, w, m; d=d, cm=false, rng=rng)
            @test nn == nv(g) == n0 + 1
            @test nl == ne(g) == e0 + m
            @test length(pos) == nv(g)
            @test degree(g, nn) == m
            @test all(v < nn for v in neighbors(g, nn))
            for v in neighbors(g, nn)
                @test w[nn, v] == w[v, nn] ≈ norm(pos[nn] - pos[v])
            end
        end
        @test nnz(w) == 2 * ne(g)

        # with rmin == rmax the new node lies exactly at that distance from the center
        r = 7.0
        add_node!(g, pos, w, m; d=d, cm=false, rmin=r, rmax=r, rng=rng)
        @test norm(pos[end]) ≈ r
        center = mean(pos)
        add_node!(g, pos, w, m; d=d, cm=true, rmin=r, rmax=r, rng=rng)
        @test norm(pos[end] - center) ≈ r
    end

    @testset "create_spa_graph: structure" begin
        for (N, m, d) in ((300, 2, 1), (500, 2, 2), (400, 4, 3))
            g, pos, w = create_spa_graph(N, m; d=d, cm=false, rng=MersenneTwister(7))
            @test nv(g) == N
            @test ne(g) == m * (m - 1) ÷ 2 + (N - m) * m
            @test !has_self_loops(g)
            @test length(pos) == N
            @test all(length(p) == d for p in pos)
            # each grown node attaches to exactly m older nodes
            @test all(count(<(i), neighbors(g, i)) == m for i in m+1:N)
            @test size(w) == (N, N)
            @test w == w'
            @test nnz(w) == 2 * ne(g)
            @test all(w[e.src, e.dst] ≈ norm(pos[e.src] - pos[e.dst]) for e in edges(g))
            # nodes lie in the shell rmin <= |x| <= rmax (cm=false), seeds up to ε jitter
            @test all(1.0 - 1e-3 * sqrt(d) <= norm(p) <= 1000.0 + 1e-9 for p in pos)
        end
        # m = 1 seeds with two connected nodes, then grows a tree
        g, _, _ = create_spa_graph(50, 1; rng=MersenneTwister(7))
        @test nv(g) == 50
        @test ne(g) == 49
        @test is_connected(g)
        # in d = 1 rand_hypersphere returns ±1; the ε jitter keeps seed nodes apart
        @test all(1:20) do s
            pos = create_spa_graph(10, 2; d=1, rng=MersenneTwister(s))[2]
            pos[1] != pos[2]
        end
    end

    @testset "create_spa_graph: reproducibility" begin
        g1, p1, w1 = create_spa_graph(400, 2; rng=MersenneTwister(8))
        g2, p2, w2 = create_spa_graph(400, 2; rng=MersenneTwister(8))
        g3, p3, w3 = create_spa_graph(400, 2; rng=MersenneTwister(9))
        @test g1 == g2 && p1 == p2 && w1 == w2
        @test p1 != p3
    end

    # Ensemble statistics of the model, used as a regression guard: any refactor
    # (even one that changes the random stream) must reproduce them.
    # Reference values: mean and std over 200 seeds, N = 1000, m = 2, d = 2,
    # αG = 2, γ = 1. cm = false (new nodes placed around the origin): the
    # original implementation; cm = true (around the center of mass, the
    # default since 2026-10-10): seeds 1:200 of the current implementation.
    @testset "create_spa_graph: ensemble statistics (cm = $cm, αA = $αA)" for (cm, αA, ref) in (
        (false, 2.0, (kmax=(57.28, 9.11), fmin=(0.4960, 0.0110), mlogL=(-0.9974, 0.0431), k1=(43.84, 13.40))),
        (false, 0.0, (kmax=(81.37, 17.08), fmin=(0.4987, 0.0106), mlogL=(0.7419, 0.0432), k1=(52.05, 24.68))),
        (false, 6.0, (kmax=(24.52, 2.67), fmin=(0.3782, 0.0083), mlogL=(-1.9079, 0.0352), k1=(18.77, 3.79))),
        (true, 2.0, (kmax=(55.32, 11.20), fmin=(0.4892, 0.0112), mlogL=(-0.9235, 0.0566), k1=(31.95, 17.53))),
        (true, 0.0, (kmax=(82.36, 17.91), fmin=(0.4984, 0.0106), mlogL=(0.7520, 0.0567), k1=(50.12, 24.85))),
        (true, 6.0, (kmax=(24.31, 3.39), fmin=(0.3740, 0.0087), mlogL=(-1.8549, 0.0451), k1=(15.91, 5.98))),
    )
        N, m, nseeds = 1000, 2, 50
        stats = map(1:nseeds) do seed
            g, _, w = create_spa_graph(N, m; αA=αA, cm=cm, rng=MersenneTwister(1000 + seed))
            k = degree(g)
            (kmax=maximum(k),                                       # hub size
                fmin=mean(k .== m),                                 # fraction of leaves
                mlogL=mean(log(w[e.src, e.dst]) for e in edges(g)), # typical edge length
                k1=k[1])                                            # degree of a seed node
        end
        for f in keys(ref)
            μ, σ = ref[f]
            @test abs(mean(getfield.(stats, f)) - μ) <= 5σ / sqrt(nseeds)
        end
    end
end

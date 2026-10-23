using TsallisNetworks.CopyRandom
using Graphs
using Random
using Test

# straightforward reference implementation, same RNG draws as the fast one
function naive(N, α; m0=2, n0=m0 + 1, link_parent=false, rng)
    g = complete_graph(n0)
    for t in (n0+1):N
        add_vertex!(g)
        if rand(rng) < α
            v = rand(rng, 1:(t-1))
            for u in collect(neighbors(g, v))
                add_edge!(g, t, u)
            end
            link_parent && add_edge!(g, t, v)
        else
            chosen = Int[]
            for _ in 1:m0
                u = rand(rng, 1:(t-1))
                while u in chosen
                    u = rand(rng, 1:(t-1))
                end
                push!(chosen, u)
                add_edge!(g, t, u)
            end
        end
    end
    return g
end

@testset "copy random model" begin
    @testset "matches naive implementation" begin
        for (α, m0, n0, lp) in ((0.5, 2, 3, false), (0.8, 3, 5, false),
            (0.3, 1, 2, true), (0.9, 4, 4, true))
            g = copy_random_graph(3000, α; m0, n0, link_parent=lp, rng=Xoshiro(7))
            h = naive(3000, α; m0, n0, link_parent=lp, rng=Xoshiro(7))
            @test g == h
            @test ne(g) == sum(degree(g)) ÷ 2
            @test all(issorted, g.fadjlist)
            @test !has_self_loops(g)
        end
    end

    @testset "limiting cases" begin
        # α = 0: every new node brings exactly m0 edges
        g = copy_random_graph(10_000, 0.0; m0=3, n0=4, rng=Xoshiro(1))
        @test ne(g) == 6 + (10_000 - 4) * 3
        @test is_connected(g)
        # α = 1 from K_n0: complete n0-partite graph, n0 distinct neighborhoods
        g = copy_random_graph(2000, 1.0; n0=4, rng=Xoshiro(2))
        @test length(unique(g.fadjlist)) == 4
    end

    @testset "Int32 labels" begin
        g = copy_random_graph(1000, 0.5; rng=Xoshiro(3), T=Int32)
        @test eltype(g) == Int32
        @test g == copy_random_graph(1000, 0.5; rng=Xoshiro(3))
    end
end

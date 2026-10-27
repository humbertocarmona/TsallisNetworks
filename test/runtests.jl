using TsallisNetworks
using DataFrames
using Test

@testset "TsallisNetworks.jl" begin
    @testset "bz2 csv round trip" begin
        df = DataFrame(a=1:3, b=[0.5, 1.5, 2.5])
        path = tempname() * ".csv.bz2"
        save_bz2_csv(path, df)
        @test load_bz2_csv(path) == df
        rm(path)
    end

    include("copy_random.jl")
    include("observables.jl")
    include("configuration_model.jl")
    include("preferential_random.jl")
    include("edge_recruitment.jl")
    include("productivity_collaboration.jl")
    include("spatial_preferential_attachment.jl")
end

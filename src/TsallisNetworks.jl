module TsallisNetworks

using CodecBzip2
using CSV
using DataFrames

include("utils.jl")

include("Observables/Observables.jl")
include("CopyRandom/CopyRandom.jl")
include("ConfigurationModel/ConfigurationModel.jl")
include("PreferentialRandom/PreferentialRandom.jl")
include("EdgeRecruitment/EdgeRecruitment.jl")
include("ProductivityCollaboration/ProductivityCollaboration.jl")
include("SpatialPreferentialAttachment/SpatialPreferentialAttachment.jl")

export CopyRandom,
    ConfigurationModel,
    EdgeRecruitment,
    Observables,
    PreferentialRandom,
    ProductivityCollaboration,
    SpatialPreferentialAttachment,
    expq,
    lnq,
    load_bz2_csv,
    save_bz2_csv

end

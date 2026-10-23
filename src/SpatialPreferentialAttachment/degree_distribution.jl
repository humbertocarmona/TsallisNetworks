using StatsBase

function degree_distribution(degs::AbstractVector; log_binning = false, nbins = 20)
	if log_binning
		i_max = log2(maximum(degs))
		i_min = log2(minimum(degs))		
		bins = 2 .^ range(i_min, i_max; length = nbins)
		x = bins[1:(end-1)] .+ diff(bins)/2
		y = StatsBase.fit(Histogram, degs, bins; closed = :left).weights
		w = diff(bins)
		y = y ./ sum(y) ./ w

		return Float64.(x), Float64.(y)
	else
		counts = countmap(degs)                 # Dict{Int,Int}
		x = sort!(collect(keys(counts)))
		ys = [counts[k] for k in x]
		w = diff(x)
        push!(w, w[end])
        ys = ys ./ sum(ys) ./ w

		return Float64.(x), Float64.(ys)
	end
end


function degree_distribution(g::Graphs.AbstractGraph; log_binning = false, nbins = 20)
	degs = Graphs.degree(g)
    return degree_distribution(degs; log_binning = log_binning, nbins = nbins)
end
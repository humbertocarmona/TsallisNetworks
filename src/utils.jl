"""
expq(x, q) = [1 + (1-q) * x]^(1/(1-q)) if [1 + (1-q) * x] >0
    q-exponential with parameter `q`. For `q == 1` it reduces to `exp(x)`.
    This method is scalar; use broadcasting `expq.(x, q)` for arrays.
"""
@inline function expq(x::Real, q::Real)
    q1 = 1 - q

    # q → 1 limit gives the standard exponential
    if iszero(q1)
        return exp(x)
    end
    base = 1 + q1 * x

    # Optional: enforce domain cut-off (negative base → 0)
    if base <= 0
        return zero(promote_type(typeof(x), typeof(q)))
    end

    return base^(1 / q1)
end

@inline function lnq(x::Real, q::Real)
    @assert x > 0 "ln_q is only defined for x > 0"

    q1 = 1 - q

    if iszero(q1)
        return log(x)
    end

    return (x^q1 - 1) / q1
end

function save_bz2_csv(path, df)
    open(Bzip2CompressorStream, path, "w") do io
        CSV.write(io, df)
    end
end

function load_bz2_csv(path)
    open(path, "r") do io
        CSV.read(Bzip2DecompressorStream(io), DataFrame)
    end
end

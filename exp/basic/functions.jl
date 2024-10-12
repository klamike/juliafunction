function muladd(x, y) # 2 -> 2
    return x .* y, x .+ y
end

function mul(x, y) # 2 -> 1
    return x .* y
end

function identity(x) # 1 -> 1
    return x
end

function double(x) # 1 -> 2
    return -x, 2*x
end
struct Monomial
    coef::Rational{Int}
    powers::Dict{DTerm,DTerm}
end

function addpower!(powers::Dict{DTerm,DTerm}, factor::DTerm, power::DTerm)
    iszero(power) && return

    if haskey(powers, factor)
        old_power = powers[factor]
        combined = if isinteger(old_power) && isinteger(power)
            old_p = Int(value(old_power))::Int
            p = Int(value(power))::Int
            DLiteral(old_p + p)
        else
            old_power + power
        end

        if iszero(combined)
            delete!(powers, factor)
        else
            powers[factor] = combined
        end
    else
        powers[factor] = power
    end

    return
end

function collect_monomial!(powers::Dict{DTerm,DTerm}, coef::Rational{Int}, factor::DTerm, power::DTerm)::Rational{Int}
    p = tryconvert(Int, power)

    if isrational(factor) && !isnothing(p)
        factor_value = value(Union{Rational,Integer}, factor)
        literal = Rational{Int}(factor_value)
        return coef * literal^p
    end

    @match factor begin
        DCall(f, args) => begin
            if f === (*)
                for arg in args
                    coef = collect_monomial!(powers, coef, arg, power)
                end
                return coef
            elseif f === (^) && length(args) == 2 && !isnothing(p)
                base, exponent = args
                if isone(p)
                    return collect_monomial!(powers, coef, base, exponent)
                end

                exponent_p = tryconvert(Int, exponent)
                if !isnothing(exponent_p)
                    return collect_monomial!(powers, coef, base, DLiteral(p * exponent_p))
                end
            end
        end
        _ => nothing
    end

    addpower!(powers, factor, power)
    return coef
end

function Monomial(term::DTerm)
    powers = Dict{DTerm,DTerm}()
    coef = collect_monomial!(powers, 1 // 1, term, DLiteral(1))
    return Monomial(coef, powers)
end

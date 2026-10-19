# Non-regression proof for a comment/docstring-ONLY cut.
#
# Julia's parser drops comments. A docstring is NOT a bare string in the AST:
# `Meta.parseall` wraps it in a `Core.var"@doc"(ln, doc, def)` call. FIELD
# docstrings, on the other hand, stay bare strings INSIDE the struct's block. So a
# normalizer that drops only bare strings misses every function and struct docstring,
# and one that drops ALL strings misses the other half: it declares equal two files
# where a CODE string changed (`error("boom")` -> `error("bang")`). Both traps were
# found while validating this script before trusting it (positive AND negative
# controls, plus a negative control on a code string).
#
# Rule: a string is a docstring only in EXPRESSION position inside a block
# (block/module/struct/let) or as the argument of `@doc`. Inside a :call a string is
# code and is kept.
#
# Usage: julia scripts/ast_equal.jl BEFORE.jl AFTER.jl

const BLOCK_HEADS = (:block, :module, :struct, :let, :toplevel)

is_atdoc(e::Expr) = e.head === :macrocall && length(e.args) == 4 &&
    string(e.args[1]) == "Core.var\"@doc\""

is_docstring_node(a) = (a isa String) ||
    (isa(a, Expr) && a.head === :string && all(x -> x isa String, a.args))

normalize(x) = x

function normalize(e::Expr)
    is_atdoc(e) && return normalize(e.args[end])   # the definition is the LAST argument
    inblock = e.head in BLOCK_HEADS
    args = Any[]
    for a in e.args
        a isa LineNumberNode && continue
        inblock && is_docstring_node(a) && continue
        push!(args, normalize(a))
    end
    return Expr(e.head, args...)
end

function parse_file(f)
    try
        Meta.parseall(read(f, String), filename=f)
    catch err
        println("PARSE FAILED $f: $err"); exit(2)
    end
end

function report(ga, gb)
    println("  top-level expressions: ", length(ga), " vs ", length(gb))
    for i in 1:min(length(ga), length(gb))
        ga[i] == gb[i] && continue
        println("  first divergence at expression ", i, ":")
        println("    before: ", first(string(ga[i]), 220))
        println("    after:  ", first(string(gb[i]), 220))
        return
    end
end

function main()
    length(ARGS) == 2 || error("usage: ast_equal.jl BEFORE.jl AFTER.jl")
    ea, eb = normalize(parse_file(ARGS[1])), normalize(parse_file(ARGS[2]))
    if ea == eb
        println("CODE IDENTICAL  ", ARGS[1], "  vs  ", ARGS[2])
        exit(0)
    end
    println("CODE MISMATCH between ", ARGS[1], " and ", ARGS[2])
    report(ea.args, eb.args)
    exit(1)
end
main()

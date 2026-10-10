# Preuve de non-regression pour une coupe de commentaires/docstrings ONLY.
#
# Le parseur de Julia jette les commentaires. Une docstring N'EST PAS une chaine nue
# dans l'AST: Meta.parseall l'enveloppe dans un appel `Core.var"@doc"(ln, doc, def)`.
# Les docstrings de CHAMP restent en revanche des chaines nues DANS LE BLOC du struct.
# Un normalisateur qui ne retire que les chaines nues rate donc toutes les docstrings
# de fonctions et de structs; un normalisateur qui retire TOUTES les chaines en rate
# une moitie: il declare identiques deux fichiers ou une chaine de CODE a change
# (`error("boom")` -> `error("bang")`). Les deux pieges ont ete trouve en validant ce
# script avant de s'en servir (controles positif ET negatif, plus un controle negatif
# sur chaine de code).
#
# Regle: une chaine n'est une docstring que si elle est en POSITION D'EXPRESSION d'un
# bloc (block/module/struct/let) ou l'argument de `@doc`. A l'interieur d'un :call, une
# chaine est du code et elle est gardee.
#
# Usage: julia scripts/ast_equal.jl AVANT.jl APRES.jl

const BLOCK_HEADS = (:block, :module, :struct, :let, :toplevel)

is_atdoc(e::Expr) = e.head === :macrocall && length(e.args) == 4 &&
    string(e.args[1]) == "Core.var\"@doc\""

is_docstring_node(a) = (a isa String) ||
    (isa(a, Expr) && a.head === :string && all(x -> x isa String, a.args))

normalize(x) = x

function normalize(e::Expr)
    is_atdoc(e) && return normalize(e.args[end])   # la definition est le DERNIER argument
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
        println("PARSE IMPOSSIBLE $f: $err"); exit(2)
    end
end

function report(ga, gb)
    println("  expressions au niveau superieur: ", length(ga), " vs ", length(gb))
    for i in 1:min(length(ga), length(gb))
        ga[i] == gb[i] && continue
        println("  1re divergence a l'expression ", i, ":")
        println("    avant: ", first(string(ga[i]), 220))
        println("    apres: ", first(string(gb[i]), 220))
        return
    end
end

function main()
    length(ARGS) == 2 || error("usage: ast_equal.jl AVANT.jl APRES.jl")
    ea, eb = normalize(parse_file(ARGS[1])), normalize(parse_file(ARGS[2]))
    if ea == eb
        println("CODE IDENTIQUE  ", ARGS[1], "  vs  ", ARGS[2])
        exit(0)
    end
    println("DESACCORD DE CODE entre ", ARGS[1], " et ", ARGS[2])
    report(ea.args, eb.args)
    exit(1)
end
main()

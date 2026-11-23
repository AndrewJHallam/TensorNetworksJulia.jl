module ExampleOutput
export write_rows

"""Write numeric/Boolean NamedTuple rows, without adding a CSV dependency."""
function write_rows(path, rows)
    isempty(rows) && throw(ArgumentError("no rows to write"))
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            keys(row) == keys(first(rows)) || error("inconsistent output columns")
            println(io, join(string.(values(row)), ','))
        end
    end
    return path
end
end

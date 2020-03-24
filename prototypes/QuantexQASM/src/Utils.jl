module Utils
export format_string_nl

"""
format_string_nl(circuit_output::String)

Simple utility to format the OpenQASM output for user readability (i.e. adds newlines).
"""
function format_string_nl(circuit_output::String)
    return replace(circuit_output, ";"=>";\n")
end

end

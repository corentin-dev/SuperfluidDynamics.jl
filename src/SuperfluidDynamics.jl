module SuperfluidDynamics

using Preferences, MPIPreferences
using DocStringExtensions

using MPI
using HDF5

using AbstractFFTs
using FFTW

using PencilArrays
using LinearAlgebra: mul!, ldiv!, transpose!

include("MPI/MPI.jl")
include("Discretization/Discretization.jl")
include("Potentials/Potentials.jl")
include("Parameters/Parameters.jl")
include("Inits/Inits.jl")
include("IO/IO.jl")
include("NumModels/NumModels.jl")
include("HVBK/HVBK.jl")

end # module

# Shared test infrastructure.
include("Shared/TestHelpers.jl")
include("Shared/MPOActionTests.jl")

# Open-boundary finite MPS/MPO tests.
include("Finite/CoreTests.jl")
include("Finite/TDVPTests.jl")
include("Finite/TEBDTests.jl")
include("Finite/ModelTests.jl")
include("Finite/MPOBridgeTests.jl")
include("Finite/ObservableTests.jl")
include("Finite/CompressionTests.jl")
include("Finite/LongRangeMPOTests.jl")
include("Finite/AllocationTests.jl")
include("PDE/VlasovTests.jl")
include("PDE/MatlabVlasovTests.jl")
include("PDE/SqrtVlasovTests.jl")

# One-site-unit-cell infinite MPS tests.
include("Infinite/NearestNeighborTests.jl")
include("Infinite/ObservableTests.jl")
include("Infinite/BondExpansionTests.jl")
include("Infinite/MPOTests.jl")
include("Infinite/LongRangeTests.jl")
include("Infinite/UnitCellTests.jl")
include("Infinite/AllocationTests.jl")

include("Shared/MatlabParityTests.jl")

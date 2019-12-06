module AdvancedMH

# Import the relevant libraries.
using Reexport
using AbstractMCMC
using Distributions
using Random
using ForwardDiff: gradient!
using DiffResults: GradientResult, value, gradient

# Import specific functions and types to use or overload.
import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples

# Define a model type. Stores the log density function and the data to
# evaluate the log density on.
"""
    DensityModel{F<:Function} <: AbstractModel

`DensityModel` wraps around a self-contained log-liklihood function `ℓπ`.

Example:

```julia
l
DensityModel
```
"""
struct DensityModel{F<:Function} <: AbstractModel
    ℓπ :: F
end

#include relevant funcationality
include("MALA.jl")
include("MH.jl")

# Exports
export MetropolisHastings, MALA, DensityModel, sample

# Define the other step functions. Returns a Transition containing
# either a new proposal (if accepted) or the previous proposal
# (if not accepted).
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::Union{MetropolisHastings, MALA},
    ::Integer,
    θ_prev::Union{Transition, Transition_w∇};
    kwargs...
)
    # Generate a new proposal.
    θ = propose(spl, model, θ_prev)

    # Calculate the log acceptance probability.
    α = ℓπ(model, θ) - ℓπ(model, θ_prev) + q(spl, θ_prev, θ) - q(spl, θ, θ_prev)

    # Decide whether to return the previous θ or the new one.
    if log(rand(rng)) < min(α, 0.0)
        return θ
    else
        return θ_prev
    end
end

# A basic chains constructor that works with the Transition struct we defined.
function bundle_samples(
    rng::AbstractRNG,
    ℓ::DensityModel,
    s::Union{MetropolisHastings, MALA},
    N::Integer,
    ts::Vector{T};
    param_names=missing,
    kwargs...
) where {ModelType<:AbstractModel, T<:AbstractTransition}
    # Turn all the transitions into a vector-of-vectors.
    vals = copy(reduce(hcat,[vcat(t.θ, t.lp) for t in ts])')

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = ["Parameter $i" for i in 1:(length(first(vals))-1)]
    end

    # Add the log density field to the parameter names.
    push!(param_names, "lp")

    # Bundle everything up and return a Chains struct.
    return Chains(vals, param_names, (internals=["lp"],))
end

end # module AdvancedMH

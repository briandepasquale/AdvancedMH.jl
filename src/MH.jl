"""
    MetropolisHastings{T, F<:Function}

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
MetropolisHastings([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
struct MetropolisHastings{T, D} <: AbstractSampler
    init_θ :: T
    proposal :: D
end

# Default constructors.
MetropolisHastings(init_θ::Real) = MetropolisHastings(init_θ, Normal(0,1))
MetropolisHastings(init_θ::Vector{<:Real}) = MetropolisHastings(init_θ, MvNormal(length(init_θ),1))


# Create a very basic Transition type, only stores the
# parameter draws and the log probability of the draw.
struct Transition{T<:Union{Vector{<:Real}, <:Real}, L<:Real} <: AbstractTransition
    θ :: T
    lp :: L
end

# Store the new draw and its log density.
Transition(model::M, θ::T) where {M<:DensityModel, T} = Transition(θ, ℓπ(model, θ))

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::MetropolisHastings) = typeof(Transition(spl.init_θ, ℓπ(model, spl.init_θ)))

# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MetropolisHastings, model::DensityModel, θ::Real) =
    Transition(model, θ + rand(spl.proposal))
propose(spl::MetropolisHastings, model::DensityModel, θ::Vector{<:Real}) =
    Transition(model, θ + rand(spl.proposal))
propose(spl::MetropolisHastings, model::DensityModel, t::Transition) = propose(spl, model, t.θ)

"""
    q(θ::Real, dist::Sampleable)
    q(θ::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(θ | θcond)`, using the proposal distribution `spl.proposal`.
"""
q(spl::MetropolisHastings, θ::Real, θcond::Real) = logpdf(spl.proposal, θ - θcond)
q(spl::MetropolisHastings, θ::Vector{<:Real}, θcond::Vector{<:Real}) = logpdf(spl.proposal, θ - θcond)
q(spl::MetropolisHastings, t1::Transition, t2::Transition) = q(spl, t1.θ, t2.θ)

# Calculate the density of the model given some parameterization.
ℓπ(model::DensityModel, θ::T) where T = model.ℓπ(θ)
ℓπ(model::DensityModel, t::Union{Transition, Transition_w∇}) = t.lp


# Define the first step! function, which is called at the
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
    rng::AbstractRNG,
    model::DensityModel,
    spl::MetropolisHastings,
    N::Integer;
    kwargs...
)
    return Transition(model, spl.init_θ)
end

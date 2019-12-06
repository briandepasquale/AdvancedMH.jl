"""
    MALA{T, F<:Function}

Fields:

- `init_θ` is the vector form of the parameters needed for the likelihood function.
- `proposal` is a function that dynamically constructs a conditional distribution.

Example:

```julia
MALA([0.0, 0.0], x -> MvNormal(x, 1.0))
````
"""
struct MALA{T, D} <: AbstractSampler
    init_θ :: T
    proposal :: D
end


# Default constructors.
MALA(init_θ::Real) = MetropolisHastings(init_θ)

# Create a very basic Transition type, only stores the
# parameter draws, the log probability of the draw and the gradient of the log probabiliyt of the draw.
struct Transition_w∇{T<:Union{Vector{<:Real}, <:Real}, L<:Real, G<:Union{Vector{<:Real}, <:Real}} <: AbstractTransition
    θ :: T
    lp :: L
    ∇ :: G
end

# Store the new draw, its log density and gradient of the log density.
Transition_w∇(model::M, θ::T) where {M<:DensityModel, T} = Transition_w∇(θ, ∂ℓπ∂θ(model, θ)...)

# Tell the interface what transition type we would like to use.
transition_type(model::DensityModel, spl::MALA) = typeof(Transition_w∇(spl.init_θ, ∂ℓπ∂θ(model, spl.init_θ)...))


# Define a function that makes a basic proposal depending on a univariate
# parameterization or a multivariate parameterization.
propose(spl::MALA, model::DensityModel, θ::Real, ∇::Vector{<:Real}) =
    Transition_w∇(model, θ + rand(spl.proposal(∇)))
propose(spl::MALA, model::DensityModel, θ::Vector{<:Real}, ∇::Vector{<:Real}) =
    Transition_w∇(model, θ + rand(spl.proposal(∇)))
propose(spl::MALA, model::DensityModel, t::Transition_w∇) = propose(spl, model, t.θ, t.∇)


"""
    q(θ::Real, dist::Sampleable)
    q(θ::Vector{<:Real}, dist::Sampleable)
    q(t1::Transition, dist::Sampleable)

Calculates the probability `q(θ | θcond)`, using the proposal distribution `spl.proposal`.
"""
q(spl::MALA, θ::Real, θcond::Real, ∇::Real) = logpdf(spl.proposal(-∇), θ - θcond)
q(spl::MALA, θ::Vector{<:Real}, θcond::Vector{<:Real}, ∇::Vector{<:Real}) = logpdf(spl.proposal(-∇), θ - θcond)
q(spl::MALA, t1::Transition_w∇, t2::Transition_w∇) = q(spl, t1.θ, t2.θ, t2.∇)

∂ℓπ∂θ(model::DensityModel, t::Transition_w∇) = (t.lp, t.∇)


"""
    ∂ℓπ∂θ(model::DensityModel, θ::T)

Efficiently returns the value and gradient of the model
"""
function ∂ℓπ∂θ(model::DensityModel, θ::T) where T
    res = GradientResult(θ)
    gradient!(res, model.ℓπ, θ)
    return (value(res), gradient(res))
end

#Define the first step! function, which is called at the
# beginning of sampling. Return the initial parameter used
# to define the sampler.
function step!(
   rng::AbstractRNG,
   model::DensityModel,
   spl::MALA,
   N::Integer;
   kwargs...
)
   return Transition_w∇(model, spl.init_θ)
end

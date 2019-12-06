using Test
using AdvancedMH
using Random
using Distributions

@testset "AdvancedMH" begin

    Random.seed!(1)
    #μ, σ, n = rand(Normal(0,1)), sqrt(rand(InverseGamma(2,3))), 30
    μ, σ, n = 5, 3, 100

    # Generate a set of data from the posterior we want to estimate.
    data = rand(Normal(μ, σ), n)

    μhat = mean(data)
    σhat = std(data)

    # Define the components of a basic model.
    insupport(θ) = θ[2] >= 0
    dist(θ) = Normal(θ[1], θ[2])
    density(θ) = insupport(θ) ? sum(logpdf.(dist(θ), data)) : -Inf

    # Construct a DensityModel.
    model = DensityModel(density)

    x0 = [0.0, 1.0]
    nsamps = 100000

    # Set up our sampler with initial parameters.
    spl = MetropolisHastings(x0)

    # Sample from the posterior.
    chain = sample(model, spl, nsamps; param_names=["μ", "σ"])

    # chn_mean ≈ dist_mean atol=atol_v
    @test mean(chain["μ"].value) ≈ μhat atol=0.1
    @test mean(chain["σ"].value) ≈ σhat atol=0.1

    τ = 0.01
    # Set up our sampler with initial parameters.
    spl_MALA = MALA(x0, x-> MvNormal(τ * 0.5 * x, τ))

    # Sample from the posterior.
    chain_MALA = sample(model, spl_MALA, nsamps; param_names=["μ", "σ"])

    @test mean(chain_MALA["μ"].value) ≈ μhat atol=0.1
    @test mean(chain_MALA["σ"].value) ≈ σhat atol=0.1

end

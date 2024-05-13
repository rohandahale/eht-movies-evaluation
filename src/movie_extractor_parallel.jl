######################################################################
# Author: Rohan Dahale, Date: 24 Jan 2024
# Based on image_extractor.jl taken from VIDA.jl github scripts
######################################################################

"""
Julia Version 1.10.2
Commit bd47eca2c8a (2024-03-01 10:14 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 112 × AMD EPYC 7B13
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
Threads: 1 default, 0 interactive, 1 GC (on 112 virtual cores)

  [c7e460c6] ArgParse v1.1.5
  [336ed68f] CSV v0.10.13
  [99d987ce] Comrade v0.9.3
  [a93c6f00] DataFrames v1.6.1
  [c27321d9] Glob v1.3.1
  [3e6eede4] OptimizationBBO v0.2.1
  [bd407f91] OptimizationCMAEvolutionStrategy v0.2.1
  [3aafef2f] OptimizationMetaheuristics v0.2.0
  [4096cdfb] VIDA v0.11.4
  [9a3f8284] Random
"""

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Distributed
@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
end

@everywhere using VIDA

using ArgParse
using CSV
using DataFrames
using Random
using Glob

@everywhere begin
    using OptimizationBBO
    using OptimizationMetaheuristics
    using OptimizationCMAEvolutionStrategy
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--input"
            help = "input hdf5 movie file to read"
            arg_type = String
            required = true
        "--output"
            help = "output csv file with extractions"
            arg_type = String
            required = true
        "--stride"
             help = "Checkpointing stride, i.e. number of steps."
             arg_type = Int
             default = 8
        "--blur"
             help = "Blur images before extracting fitted model parameters"
             arg_type = Float64
             default = 0.0
        "--restart"
            help = "Tells the sampler to read in the old frames and restart the run."
            action = :store_true
        "--template"
            help = "Parses sting with models separated by spaces. For instance to\n"*
                   "run a model with 1 m=(1,4) m-ring, 2 gaussian and a stretched disk\n"*
                   "one could do `--template mring_1_4 gauss_2 disk_1`. The current model\n"*
                   "options are: \n"*
                   "  - `[stretch]mring_n_m`: adds a [stretched] m-ring of order `n` `m` thickenss and azimuth resp.\n"*
                   "  - `gauss_n`           : add `n` asymmetric gaussian components to the template\n"*
                   "  - `[stretch]disk_n    : add `n` [stretched] flat top disks to the template\n"
            action = :store_arg
            nargs = '*'
            arg_type = String
            required = true
        "--seed"
            help = "Random seed for initial positions in extract"
            arg_type = Int
            default = 42
    end
    return parse_args(s)
end

function main()
    #Parse the command line arguments
    parsed_args = parse_commandline()
    #Assign the arguments to specific variables
    file = parsed_args["input"]
    output = parsed_args["output"]
    seed = parsed_args["seed"]
    stride = parsed_args["stride"]
    blur = parsed_args["blur"]
    templates = parsed_args["template"]
    @info "Template types $templates"
    restart = parsed_args["restart"]
    
    out_name = output
    if !isfile(out_name)
        println("Using options: ")
        println("output name: $out_name, ")
        println("random seed $seed")
        println("Checkpoint stride $stride")
        println("Blurring Gaussian kernel width in µas: $blur")
        println("Starting fit")

        main_sub(file, out_name,
                 templates,
                 seed,
                 restart, stride, blur)
        println("Done! Check $out_name for summary")
    else
        println("$out_name already exists!")
    end
    
    
    return 0
end


function make_initial_templates(templates...)
    res = map(make_initial_template, templates)
    syms = ntuple(i->Symbol(:model_, i), length(res))
    templates = NamedTuple{syms}(getindex.(res, 1))
    lowers    = NamedTuple{syms}(getindex.(res, 2))
    uppers    = NamedTuple{syms}(getindex.(res, 3))

    templates_const = merge(templates, (c = x->(x.floor*Constant(μas2rad(150.0))),))
    lowers_const = merge(lowers, (c = (floor = 1e-6,),))
    uppers_const = merge(uppers, (c = (floor = 100.0,),))

    syms_const = (syms..., :c)

    temp = let t = templates_const, s=syms_const
        x->begin
            sum(s) do si
                getproperty(t, si)(getproperty(x, si))
            end
        end
    end
    return temp, lowers_const, uppers_const
end



function make_initial_template(template)
    if occursin("mring", template)
        stretch = occursin("stretch", template)
        type = parse.(Int, split(template, "_")[2:3])
        @info "Template includes a m-ring of order $(type) with stretch $stretch"
        return make_initial_template_mring(type[1], type[2], stretch)
    elseif occursin("gauss", template)
        ngauss = parse.(Int, split(template, "_")[end])
        @info "Template includes $ngauss gaussians"
        return make_initial_template_gauss(ngauss)
    elseif occursin("disk", template)
        ndisk = parse.(Int, split(template, "_")[end])
        stretch = occursin("stretch", template)
        @info "Template includes $ndisk disks with stretch $stretch"
        return make_initial_template_disk(ndisk, stretch)
    else
        @error "Template $template not available"
    end
end



function make_initial_template_gauss(n::Int)
    gauss = @inline function (θ)
        mapreduce(+, 1:n) do i
            modify(Gaussian(),
                Stretch(θ.σ[i], θ.σ[i]),
                #Rotate(θ.ξ[i]),
                Shift(θ.x0[i], θ.y0[i])
            )
        end
    end
    lower = (σ=ntuple(_-> μas2rad(0.5), n),
             #τ=ntuple(_->0.001, n),
             #ξ=ntuple(_->-π/2, n),
             x0=ntuple(_->-μas2rad(60.0), n),
             y0=ntuple(_->-μas2rad(60.0), n),
             )
    upper = (σ=ntuple(_-> μas2rad(100.0), n),
             #τ=ntuple(_->0.5, n),
             #ξ=ntuple(_->π/2, n),
             x0=ntuple(_->μas2rad(60.0), n),
             y0=ntuple(_->μas2rad(60.0), n),
             )

    return gauss, lower, upper
end

function make_initial_template_disk(n::Int, stretch)
    if stretch
        disk = @inline function (θ)
            mapreduce(+, 1:n) do i
                mod = sqrt(1-θ.τ[i])
                modify(GaussDisk(θ.σ[i]/θ.r0[i]),
                        Stretch(θ.r0[i]*mod, θ.r0[i]/mod),
                          Rotate(θ.ξ[i]),
                          Shift(θ.x0[i], θ.y0[i])
                         )
            end
        end
        lower = (
                 r0=ntuple(_-> μas2rad(0.5), n),
                 σ =ntuple(_->μas2rad(0.1), n),
                 τ=ntuple(_->0.001, n),
                 ξ=ntuple(_->-π/2, n),
                 x0=ntuple(_->-μas2rad(60.0), n),
                 y0=ntuple(_->-μas2rad(60.0), n),
                )
        upper = (
                 r0=ntuple(_->μas2rad(100.0), n),
                 σ =ntuple(_->μas2rad(20.0), n),
                 τ=ntuple(_->0.5, n),
                 ξ=ntuple(_->π/2, n),
                 x0=ntuple(_->μas2rad(60.0), n),
                 y0=ntuple(_->μas2rad(60.0), n),
                )
    else
        disk = @inline function (θ)
            mapreduce(+, 1:n) do i
                modify(GaussDisk(θ.σ[i]/θ.r0[i]),
                          Stretch(θ.r0[i], θ.r0[i]),
                          Shift(θ.x0[i], θ.y0[i])
                         )
            end
        end
        lower = (
                 r0=ntuple(_-> μas2rad(0.5), n),
                 σ =ntuple(_->μas2rad(0.1), n),
                 x0=ntuple(_->-μas2rad(60.0), n),
                 y0=ntuple(_->-μas2rad(60.0), n),
                )
        upper = (
                 r0=ntuple(_-> μas2rad(100.0), n),
                 σ =ntuple(_->μas2rad(20.0), n),
                 x0=ntuple(_->μas2rad(60.0), n),
                 y0=ntuple(_->μas2rad(60.0), n),
                )
    end
    return disk, lower, upper
end

function make_initial_template_mring(N::Int, M::Int, stretch)

    lower = (
             r0 = μas2rad(10.0),
             σ0 = μas2rad(1.0),
             σ  = ntuple(_->μas2rad(0.0), N),
             ξσ = ntuple(_->-1π, N),
             s  = ntuple(_->0.0, M),
             ξs = ntuple(_->-1π, M),
             x0 = -μas2rad(60.0),
             y0 = -μas2rad(60.0)
             )

    upper = (
             r0 = μas2rad(35.0),
             σ0 = μas2rad(20.0),
             σ  = ntuple(_->μas2rad(5.0), N),
             ξσ = ntuple(_->1π, N),
             s  = ntuple(_->0.999, M),
             ξs = ntuple(_->1π, M),
             x0 = μas2rad(60.0),
             y0 = μas2rad(60.0)
             )


    if stretch
        mring = @inline function (θ)
            mod = sqrt(1-θ.τ)
            return modify(CosineRing(θ.σ0/θ.r0, θ.σ./θ.r0, θ.ξσ .- θ.ξτ, θ.s, θ.ξs .- θ.ξτ),
                            Stretch(θ.r0*mod, θ.r0/mod),
                            Rotate(θ.ξτ),
                            Shift(θ.x0, θ.y0)
                         )
        end
        lower = merge(lower, (τ = 0.001, ξτ = -π/2))
        upper = merge(upper, (τ = 0.5, ξτ = π/2))
    else
        mring = @inline function (θ)
            return modify(CosineRing(θ.σ0/θ.r0, θ.σ./θ.r0, θ.ξσ, θ.s, θ.ξs),
                            Stretch(θ.r0),
                            Shift(θ.x0, θ.y0)
                         )
        end
    end
    return mring, lower, upper
end

flattenvals(x::NamedTuple) = flattenvals(values(x)...)
flattenvals(x, y...) = (flattenvals(x)..., flattenvals(y...)...)
flattenvals(x::Tuple) = flattenvals(x...)
flattenvals(x) = (x,)
flattenvals() = ()

function flattenkeys(nt::NamedTuple{N, T}) where {N, T<:Tuple}
    k = keys(nt)
    v = values(nt)
    t  = ntuple(i->combine_symbol(k[i], v[i]), length(k))
    flattenvals(t)
end

combine_symbol(a::Symbol, ::Number) = a
combine_symbol(a::Symbol, b::Symbol) = Symbol(a, "_", b)
combine_symbol(a::Symbol, ::NTuple{N, <:Real}) where {N} = ntuple(i->Symbol(a, "_", i), N)
function combine_symbol(a::Symbol, b::NamedTuple{N, T}) where {N, T}
    k = keys(b)
    v = values(b)
    ntuple(i->combine_symbol(Symbol(a, "_", k[i]), v[i]), length(k))
end

function create_initial_df(times, names, restart, out_name)
    start_indx = 1
    ntimes = length(times)
    df = DataFrame()
    if !restart
      #we want the keynames to match the model parameters
      for i in eachindex(names)
        insertcols!(df, ncol(df)+1, names[i] => zeros(ntimes))
      end
      #fill the data frame with some likely pertinent information
      df[:, :divmin] = zeros(ntimes)
      df[:, :time] =  times
    else
      df = DataFrame(CSV.File(out_name))
      start_indx = findfirst(isequal(0.0), df[:,1])
    end
    return df, start_indx
end

@everywhere function fit_template(frame, time, template, lower, upper, blur)
    println("Extracting frame at $time UT")
    image = frame
    if blur > 0.0
        image = VIDA.blur(image, blur) # INI: blur the image with Gaussian kernel of given fwhm
    end
    rimage = VIDA.regrid(image, μas2rad(200.0), μas2rad(200.0), 64, 64)
    cimage = VIDA.clipimage(0.0,rimage)
    div = VIDA.NxCorr(cimage)
    t = @elapsed begin
        prob = VIDAProblem(div, template, lower, upper)
        xopt,  θ, divmin = vida(prob, BBO_adaptive_de_rand_1_bin(); maxiters=100_000)
        #xopt2, θ, divmin = vida(prob, CMAEvolutionStrategyOpt(); init_params=xopt, maxiters=100_000)
        #xopt2, θ, divmin = vida(prob, ECA(); init_params=xopt, use_initial=true, maxiters=10_000)
        xopt2, θ, divmin = vida(prob, ECA(;options=Options(f_calls_limit = 10_000, f_tol = 1e-5)); init_params=xopt, use_initial=true)
    end
    println("This took $t seconds")
    println("The minimum divergence is $divmin")
    return xopt2, θ, divmin
end


function main_sub(file, out_name,
                  templates,
                  seed,
                  restart, stride, blur)

    #Define the template we want to use and the var bounds
    model, lower, upper = make_initial_templates(templates...)

    # get the flat keys that we will save everything to
    k = flattenkeys(lower)

    #Load movie and get times for frames
    movie = load_hdf5(string(file))
    times = get_times(movie)
    
    # Setup DataFrame
    df, start_indx = create_initial_df(times, k, restart, out_name)

    # Now fit the files
    indexpart = Iterators.partition(start_indx:length(times), stride)

    for ii in indexpart
        results = pmap(times[ii]) do t
                fit_template(get_image(movie, t), t, model, lower, upper, blur)
        end

        df[ii,1:length(k)] .= reduce(hcat, collect.(flattenvals.(first.(results))))'
        df[ii,end-1] = last.(results)
        df[ii,end] = times[ii]
        # save the file
        println("Checkpointing $(ii)")
        CSV.write(out_name, df)
    end
    #save the file
    CSV.write(out_name, df)

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

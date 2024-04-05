using Pkg
Pkg.activate(@__DIR__)

using VIDA
using ArgParse
using CSV
using DataFrames
using Random
using OptimizationBBO
using OptimizationMetaheuristics
using OptimizationCMAEvolutionStrategy


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--in"
            help = "(average frame)fits file to read"
            arg_type = String
            required = true
        "--out"
            help = "name of output csv file with extractions"
            arg_type = String
            default = "ring.csv"
    end
    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    fitsfile = parsed_args["in"]
    out_name = parsed_args["out"]
    println("file: $fitsfile, ")
    println("output name: $out_name, ")
    println("Starting fit")

    main_sub(fitsfile, out_name)
    println("Done! Check $out_name for summary")
    return 0
end

function find_ring_center(img::IntensityMap{<:Real})
    ring(x) = SlashedGaussianRing(x.r0, x.σ, x.s, x.ξ, x.x0, x.y0) +
              modify(Gaussian(), Stretch(x.σg), Shift(x.xg, x.yg), Renormalize(x.fg)) +
              x.f0*Constant(fieldofview(img).X)
    lower = (r0 = μas2rad(15.0), σ = μas2rad(1.0),
             s = 0.001, ξ = -1π,
             x0 = -μas2rad(30.0), y0 = -μas2rad(30.0),
             σg = μas2rad(30.0),
             xg = -fieldofview(img).X/4,
             yg = -fieldofview(img).Y/4,
             fg = 1e-6, f0 = 1e-6
             )
    upper = (r0 = μas2rad(40.0), σ = μas2rad(10.0),
             s = 0.999, ξ = 1π,
             x0 = μas2rad(30.0), y0 = μas2rad(30.0),
             σg = fieldofview(img).X/2,
             xg = fieldofview(img).X/4,
             yg = fieldofview(img).Y/4,
             fg = 20.0, f0=10.0
             )

    div = VIDA.LeastSquares(img)
    prob = VIDAProblem(div, ring, lower, upper)
    xopt,  θ, divmin = vida(prob, BBO_adaptive_de_rand_1_bin(); maxiters=50_000)
    xopt2, θ, divmin = vida(prob, CMAEvolutionStrategyOpt(); init_params=xopt, maxiters=50_000)

    return xopt2, divmin
end


function fit_template(file)
    println("Extracting $file")
    image = load_image(string(file))
    rimage = VIDA.regrid(image, μas2rad(200.0), μas2rad(200.0), 64, 64)
    cimage = VIDA.clipimage(0.0,rimage)
    t = @elapsed begin
        xopt, divmin = find_ring_center(cimage)
    end
    println("This took $t seconds")
    println("The minimum divergence is $divmin")
    return xopt
end

function main_sub(fitsfile, out_name)

    xopt = fit_template(fitsfile)
    df = DataFrame()
    names = ["r0", "x0", "y0"]

    for i in eachindex(names)
        insertcols!(df, ncol(df)+1, names[i] => zeros(1))
    end

    df[1, "r0"]=xopt.r0
    df[1, "x0"]=xopt.x0
    df[1, "y0"]=xopt.y0

    CSV.write(out_name, df)
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

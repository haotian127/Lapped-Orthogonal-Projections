function approx_errors(DVEC::Vector{Vector{Float64}}; frac::Float64 = 0.50, ϵ = 0.3)
    plot(xaxis = "Fraction of Coefficients Retained",
            yaxis = "Relative Approximation Error")
    T = ["eigenbasis-L", "HGLET", "LP-HGLET(ϵ=$(ϵ))", "Haar", "Walsh",
         "GHWT_c2f", "GHWT_f2c", "eGHWT", "VM-NGWP", "LP-NGWP(ϵ=$(ϵ))"]
    L = [(:dot, :red), (:solid, :teal), (:solid, :red), (:dashdot, :orange),
         (:dashdot, :pink), (:solid, :gray), (:solid, :green), (:solid, :blue),
         (:solid, :black), (:solid, :orange)]
    for i = 1:length(DVEC)
        if i in [1, 4, 5]
            continue
        end
        dvec = DVEC[i]
        N = length(dvec)
        dvec_norm = norm(dvec,2)
        dvec_sort = sort(dvec.^2) # the smallest first
        er = max.(sqrt.(reverse(cumsum(dvec_sort)))/dvec_norm, 1e-12)
        p = Int64(floor(frac*N)) + 1 # upper limit
        plot!(frac*(0:(p-1))/(p-1), er[1:p], yaxis=:log, xlims = (0.,frac),
                label = T[i], line = L[i], linewidth = 2, grid = false)
    end
end

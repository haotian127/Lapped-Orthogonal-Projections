function approx_errors(
    DVEC::Vector{Vector{Float64}};
    frac::Float64 = 0.50,
    ϵh = 0.4,
    ϵn = 0.2
)
    plot(
        xaxis = "Fraction of Coefficients Retained",
        yaxis = "Relative Approximation Error",
    )
    T = [
        "eigenbasis-L",
        "HGLET",
        "LP-HGLET(ϵ=$(ϵh))",
        "Haar",
        "Walsh",
        "GHWT_c2f",
        "GHWT_f2c",
        "eGHWT",
        "VM-NGWP",
        "LP-NGWP(ϵ=$(ϵn))",
    ]
    L = [
        (:dot, :red),
        (:solid, :teal),
        (:solid, :red),
        (:dashdot, :orange),
        (:dashdot, :pink),
        (:solid, :gray),
        (:solid, :green),
        (:solid, :blue),
        (:solid, :black),
        (:solid, :orange),
    ]
    N = length(DVEC[1])
    for i = 1:length(DVEC)
        if i in [1, 4, 5, 6, 7, 8]
            continue
        end
        dvec = DVEC[i]
        er = dvec2err(dvec)
        p = Int64(floor(frac * N)) + 1 # upper limit
        plot!(
            frac * (0:(p-1)) / (p - 1),
            er[1:p],
            yaxis = :log,
            xlims = (0.0, frac),
            label = T[i],
            line = L[i],
            linewidth = 2,
            grid = false,
        )
    end
end


function approx_curves_now(
    G_Sig,
    Gstar_Sig,
    GP,
    GP_dual,
    VM_NGWP,
    LP_NGWP;
    frac = 0.5,
    ϵh = 0.4,
    ϵn = 0.2
)
    ############# VM_NGWP
    dmatrix_VM = ngwp_analysis(G_Sig, VM_NGWP)
    dvec_vm_ngwp, BS_vm_ngwp = ngwp_bestbasis(dmatrix_VM, GP_dual)
    ############# LP_NGWP
    dmatrix_LP = ngwp_analysis(G_Sig, LP_NGWP)
    dvec_lp_ngwp, BS_lp_ngwp = ngwp_bestbasis(dmatrix_LP, GP_dual)
    ############# Laplacian L
    dvec_Laplacian = 𝚽' * G_Sig.f
    ############# HGLET
    GP = partition_tree_fiedler(G_Sig)
    dmatrixH, _, _ = HGLET_Analysis_All(G_Sig, GP)
    dvec_hglet, BS_hglet, _ =
        HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, costfun = 1)
    ############# LP-HGLET
    dmatrixlpH, _ = LPHGLET_Analysis_All(G_Sig, GP; ϵ = ϵh)
    dvec_lphglet, BS_lphglet, _ =
        HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixlpH, costfun = 1)
    ############# GHWT dictionaries
    dmatrix = ghwt_analysis!(G_Sig, GP = GP)
    ############# Haar
    BS_haar = bs_haar(GP)
    dvec_haar = dmatrix2dvec(dmatrix, GP, BS_haar)
    ############# Walsh
    BS_walsh = bs_walsh(GP)
    dvec_walsh = dmatrix2dvec(dmatrix, GP, BS_walsh)
    ############# GHWT_c2f
    dvec_c2f, BS_c2f = ghwt_c2f_bestbasis(dmatrix, GP)
    ############# GHWT_f2c
    dvec_f2c, BS_f2c = ghwt_f2c_bestbasis(dmatrix, GP)
    ############# eGHWT
    dvec_eghwt, BS_eghwt = ghwt_tf_bestbasis(dmatrix, GP)
    DVEC = [
        dvec_Laplacian[:],
        dvec_hglet[:],
        dvec_lphglet[:],
        dvec_haar[:],
        dvec_walsh[:],
        dvec_c2f[:],
        dvec_f2c[:],
        dvec_eghwt[:],
        dvec_vm_ngwp[:],
        dvec_lp_ngwp[:],
    ]
    approx_errors(DVEC; frac = frac, ϵh = ϵh, ϵn = ϵn)
    plt = plot!(xguidefontsize = 14, yguidefontsize = 14, legendfontsize = 11)
    return plt
end

function dvec2err(dvec)
    dvec = dvec[:]
    dvec_norm = norm(dvec, 2)
    dvec_sort = sort(dvec .^ 2) # the smallest first
    err = max.(sqrt.(reverse(cumsum(dvec_sort))) / dvec_norm, 1e-12)
    return err
end

function export_lp_err_curves(
    G_Sig,
    Gstar_Sig,
    GP,
    GP_dual;
    ϵs = [x for x = 0.05:0.05:0.95],
)
    N = size(G_Sig.f, 1)
    num_curves = length(ϵs)
    err_curves_lphglet = zeros(N, num_curves)
    for (i, ϵ) in enumerate(ϵs)
        dmatrixlpH = LPHGLET_Analysis_All(G_Sig, GP; ϵ = ϵ)[1]
        dvec = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixlpH, costfun = 1)[1]
        err_curves_lphglet[:, i] = dvec2err(dvec)
    end

    err_curves_lpngwp = zeros(N, num_curves)
    for (i, ϵ) in enumerate(ϵs)
        dmatrix_LP = lp_ngwp_analysis(G_Sig, 𝚽, Gstar_Sig.W, GP_dual; ϵ = ϵ)
        dvec = ngwp_bestbasis(dmatrix_LP, GP_dual)[1]
        err_curves_lpngwp[:, i] = dvec2err(dvec)
    end

    return err_curves_lphglet, err_curves_lpngwp
end

cd(@__DIR__); include("helper.jl")
using MultiscaleGraphSignalTransforms, Plots, LightGraphs, JLD, MAT
barbara = JLD.load("datasets/barbara_gray_matrix.jld", "barbara")

## Build weighted graph
G, L, X = SunFlowerGraph(N = 400); N = nv(G)
W = 1.0 * adjacency_matrix(G)

Q = incidence_matrix(G; oriented = true)
edge_weight = [e.weight for e in edges(G)]

## eigenvectors of L(G)
𝛌, 𝚽 = eigen(Matrix(L))
standardize_eigenvectors!(𝚽)

## Build Dual Graph by DAG metric
distDAG = eigDAG_Distance(𝚽, Q, N; edge_weight = edge_weight)
Gstar_Sig = dualgraph(distDAG)
G_Sig = GraphSig(W, xy = X)
GP_dual = partition_tree_fiedler(Gstar_Sig; swapRegion = false)
jmax = size(GP_dual.rs, 2) - 1  # zero-indexed

@time VM_NGWP = vm_ngwp(𝚽, GP_dual)
@time LP_NGWP = lp_ngwp(𝚽, Gstar_Sig.W, GP_dual; ϵ = 0.3)

## barbara eye graph signal
f = matread("datasets/sunflower_barbara_voronoi.mat")["f_eye_voronoi"]
f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
scatter_gplot(X; marker = f, ms = LinRange(4.0, 14.0, N), c = :greys);
plt = plot!(xlim = [-1.2,1.2], ylim = [-1.2,1.2], frame = :none)
savefig(plt, "figs/SunFlower_barbara_feye.png")

## barbara eye relative l2 approximation error by various methods
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
dvec_hglet, BS_hglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, costfun = 1)
############# LP-HGLET
dmatrixlpH, _ = LPHGLET_Analysis_All(G_Sig, GP; ϵ = 0.3)
dvec_lphglet, BS_lphglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixlpH, costfun = 1)
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
DVEC = [dvec_Laplacian[:], dvec_hglet[:], dvec_lphglet[:], dvec_haar[:], dvec_walsh[:],
        dvec_c2f[:], dvec_f2c[:], dvec_eghwt[:], dvec_vm_ngwp[:], dvec_lp_ngwp[:]]
approx_errors(DVEC; frac = 0.5);
plt = plot!(xguidefontsize = 14, yguidefontsize = 14, legendfontsize = 11)
savefig(plt, "figs/SunFlower_barbara_feye_DAG_approx.png")


## top 16 VM-NGWP
important_idx = sortperm(dvec_vm_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_vm_ngwp.levlist[important_idx[i]]
    w = VM_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_eye_DAG_VM_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-NGWP
important_idx = sortperm(dvec_lp_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lp_ngwp.levlist[important_idx[i]]
    w = LP_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_eye_DAG_LP_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 HGLET
important_idx = sortperm(dvec_hglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_hglet.levlist[important_idx[i]]
    w, _ = HGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                           GP, BS_hglet, G_Sig, method = :L)
    j, k, l = HGLET_jkl(GP, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_eye_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-HGLET
important_idx = sortperm(dvec_lphglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lphglet.levlist[important_idx[i]]
    w, _ = LPHGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                             GP, BS_lphglet, G_Sig; method = :L, ϵ = 0.3)
    j, k, l = HGLET_jkl(GP, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_eye_LP_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end




################################################
## barbara eye graph signal
f = matread("datasets/sunflower_barbara_voronoi.mat")["f_trouser_voronoi"]
f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
scatter_gplot(X; marker = f, ms = LinRange(4.0, 14.0, N), c = :greys);
plt = plot!(xlim = [-1.2,1.2], ylim = [-1.2,1.2], frame = :none)
savefig(plt, "figs/SunFlower_barbara_ftrouser.png")

## barbara trouser relative l2 approximation error by various methods
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
dvec_hglet, BS_hglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, costfun = 1)
############# LP-HGLET
dmatrixlpH, _ = LPHGLET_Analysis_All(G_Sig, GP; ϵ = 0.3)
dvec_lphglet, BS_lphglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixlpH, costfun = 1)
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
DVEC = [dvec_Laplacian[:], dvec_hglet[:], dvec_lphglet[:], dvec_haar[:], dvec_walsh[:],
        dvec_c2f[:], dvec_f2c[:], dvec_eghwt[:], dvec_vm_ngwp[:], dvec_lp_ngwp[:]]
approx_errors(DVEC; frac = 0.5);
plt = plot!(xguidefontsize = 14, yguidefontsize = 14, legendfontsize = 11)
savefig(plt, "figs/SunFlower_barbara_ftrouser_DAG_approx.png")


## top 16 VM-NGWP
important_idx = sortperm(dvec_vm_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_vm_ngwp.levlist[important_idx[i]]
    w = VM_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_trouser_DAG_VM_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-NGWP
important_idx = sortperm(dvec_lp_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lp_ngwp.levlist[important_idx[i]]
    w = LP_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_trouser_DAG_LP_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 HGLET
important_idx = sortperm(dvec_hglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_hglet.levlist[important_idx[i]]
    w, _ = HGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                           GP, BS_hglet, G_Sig, method = :L)
    j, k, l = HGLET_jkl(GP, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_trouser_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-HGLET
important_idx = sortperm(dvec_lphglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lphglet.levlist[important_idx[i]]
    w, _ = LPHGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                             GP, BS_lphglet, G_Sig; method = :L, ϵ = 0.3)
    j, k, l = HGLET_jkl(GP, dr, dc)
    scatter_gplot(X; marker = w, ms = LinRange(4.0, 14.0, N), c = :greys)
    plt = plot!(xlim = [-1.2, 1.2], ylim = [-1.2, 1.2], frame = :none,
                cbar = false, clims = (-0.15, 0.15))
    savefig(plt, "figs/SunFlower_barb_trouser_LP_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

cd(@__DIR__); include("helper.jl")
using MultiscaleGraphSignalTransforms, JLD, Plots, LightGraphs, Distances

## Build weighted toronto street network graph
G = loadgraph("datasets/new_toronto_graph.lgz"); N = nv(G)
X = load("datasets/new_toronto.jld", "xy")
dist_X = pairwise(Euclidean(), X; dims = 1)
A = 1.0 .* adjacency_matrix(G)
W = zeros(N, N); W[A .> 0] = 1 ./ dist_X[A .> 0]; W = A .* W
Q = incidence_matrix(G; oriented = true)
edge_weight = 1 ./ sqrt.(sum((Q' * X).^2, dims = 2)[:])

## eigenvectors of L(G)
deg = sum(W, dims = 1)[:]  # weighted degree vector
L = diagm(deg) - W
𝛌, 𝚽 = eigen(L)
standardize_eigenvectors!(𝚽)

## Build Dual Graph by DAG metric
distDAG = eigDAG_Distance(𝚽, Q, N; edge_weight = edge_weight) #52.375477 seconds
Gstar_Sig = dualgraph(distDAG)
G_Sig = GraphSig(A, xy = X); G_Sig = Adj2InvEuc(G_Sig)
GP_dual = partition_tree_fiedler(Gstar_Sig; swapRegion = false)
jmax = size(GP_dual.rs, 2) - 1  # zero-indexed

@time VM_NGWP = vm_ngwp(𝚽, GP_dual)
@time LP_NGWP = lp_ngwp(𝚽, Gstar_Sig.W, GP_dual; ϵ = 0.3)

##
f = zeros(N); for i in 1:N; f[i] = length(findall(dist_X[:,i] .< 1/minimum(edge_weight))); end
f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
gplot(A, X; width=1);
signal_plt = scatter_gplot!(X; marker = f, plotOrder = :s2l, ms = 3)
savefig(signal_plt, "figs/Toronto_fdensity.png")

##
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
savefig(plt, "figs/Toronto_fdensity_DAG_approx.png")

## top 16 VM-NGWP
important_idx = sortperm(dvec_vm_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_vm_ngwp.levlist[important_idx[i]]
    w = VM_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fdensity_DAG_VM_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-NGWP
important_idx = sortperm(dvec_lp_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lp_ngwp.levlist[important_idx[i]]
    w = LP_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fdensity_DAG_LP_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 HGLET
important_idx = sortperm(dvec_hglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_hglet.levlist[important_idx[i]]
    w, _ = HGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                           GP, BS_hglet, G_Sig, method = :L)
    j, k, l = HGLET_jkl(GP, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fdensity_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-HGLET
important_idx = sortperm(dvec_lphglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lphglet.levlist[important_idx[i]]
    w, _ = LPHGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                             GP, BS_lphglet, G_Sig; method = :L, ϵ = 0.3)
    j, k, l = HGLET_jkl(GP, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fdensity_LP_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end




##
f = load("datasets/new_toronto.jld", "fp")
f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
gplot(A, X; width=1)
signal_plt = scatter_gplot!(X; marker = f, plotOrder = :s2l, ms = 3)
savefig(signal_plt, "figs/Toronto_fp.png")

##
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
savefig(plt, "figs/Toronto_fp_DAG_approx.png")

## top 16 VM-NGWP
important_idx = sortperm(dvec_vm_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_vm_ngwp.levlist[important_idx[i]]
    w = VM_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fp_DAG_VM_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-NGWP
important_idx = sortperm(dvec_lp_ngwp[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lp_ngwp.levlist[important_idx[i]]
    w = LP_NGWP[dr, dc, :]
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fp_DAG_LP_NGWP_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 HGLET
important_idx = sortperm(dvec_hglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_hglet.levlist[important_idx[i]]
    w, _ = HGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                           GP, BS_hglet, G_Sig, method = :L)
    j, k, l = HGLET_jkl(GP, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fp_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

## top 16 LP-HGLET
important_idx = sortperm(dvec_lphglet[:].^2; rev = true)
for i in 1:16
    dr, dc = BS_lphglet.levlist[important_idx[i]]
    w, _ = LPHGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                             GP, BS_lphglet, G_Sig; method = :L, ϵ = 0.3)
    j, k, l = HGLET_jkl(GP, dr, dc)
    gplot(A, X; width=1)
    scatter_gplot!(X; marker = w, plotOrder = :s2l, ms = 3)
    plt = plot!(cbar = false, clims = (-0.075,0.075))
    savefig(plt, "figs/Toronto_fp_LP_HGLET_ibv$(lpad(i,2,"0"))_j$(j)_k$(k)_l$(l).png")
end

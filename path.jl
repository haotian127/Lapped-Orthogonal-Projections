cd(@__DIR__); include("helper.jl")
using MultiscaleGraphSignalTransforms, Plots, LightGraphs, JLD, MarketData, Statistics
import WaveletsExt: wiggle

## Build a path graph
N = 253
G = path_graph(N)
X = zeros(N, 2); X[:, 1] = 1:N;
W = 1.0 * adjacency_matrix(G)
L = laplacian_matrix(G)

## eigenvectors of L(G)
𝛌, 𝚽 = eigen(Matrix(L))
standardize_eigenvectors!(𝚽)

## Build Dual Graph
Gstar_Sig = GraphSig(W, xy = X)
G_Sig = GraphSig(W, xy = X)
GP_dual = partition_tree_fiedler(Gstar_Sig; swapRegion = false)
jmax = size(GP_dual.rs, 2) - 1  # zero-indexed

@time VM_NGWP = vm_ngwp(𝚽, GP_dual)
@time LP_NGWP = lp_ngwp(𝚽, Gstar_Sig.W, GP_dual; ϵ = 0.3)

## graph signal: (:O, :High), (:SPY, :High), (:AMZN, :Volume)
start = DateTime(2020, 1, 1)
stop = DateTime(2021, 1, 1)
dataset = yahoo(:AMZN, YahooOpt(period1 = start, period2 = stop))
f = values(dataset[:Volume])
f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
plt = plot(f, lw = 2, c = :black, legend = false, frame = :box, title = "SPY high with mean 0")
# savefig(plt, "figs/Path328_AAPL_Volume_signal.png")

##
############# VM_NGWP
dmatrix_VM = ngwp_analysis(G_Sig, VM_NGWP)
dvec_vm_ngwp, BS_vm_ngwp = ngwp_bestbasis(dmatrix_VM, GP_dual)
############# LP_NGWP
dmatrix_LP = ngwp_analysis(G_Sig, LP_NGWP)
dvec_lp_ngwp, BS_lp_ngwp = ngwp_bestbasis(dmatrix_LP, GP_dual)
############# HGLET
GP = partition_tree_fiedler(G_Sig)
dmatrixH, _, _ = HGLET_Analysis_All(G_Sig, GP)
dvec_hglet, BS_hglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, costfun = 1)
############# LP-HGLET
dmatrixlpH, _ = LPHGLET_Analysis_All(G_Sig, GP; ϵ = 0.3)
dvec_lphglet, BS_lphglet, _ = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixlpH, costfun = 1)

#############
plt = approx_curves_now(G_Sig, GP, GP_dual, VM_NGWP, LP_NGWP; frac = 0.3)
# savefig(plt, "figs/Path328_AAPL_Volume_approx.png")


## top VM-NGWP
important_idx = sortperm(dvec_vm_ngwp[:].^2; rev = true)
plot(layout = Plots.grid(6, 8), size = (1200, 600))
for i in 1:48
    dr, dc = BS_vm_ngwp.levlist[important_idx[i]]
    w = VM_NGWP[dr, dc, :]
    w .*= (maximum(w) > -minimum(w)) * 2 - 1
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    plot!(w, lw = 1, c = :black, legend = false, frame = :none, subplot = i,
          ylim = [minimum(w) * 1.5, maximum(w) * 1.5])
    # title = "ψ^($(j))_{$(k), $(l)}", titlefontsize = 6,
end
current()

## top LP-NGWP
important_idx = sortperm(dvec_lp_ngwp[:].^2; rev = true)
plot(layout = Plots.grid(6, 8), size = (1200, 600))
for i in 1:48
    dr, dc = BS_lp_ngwp.levlist[important_idx[i]]
    w = LP_NGWP[dr, dc, :]
    w .*= (maximum(w) > -minimum(w)) * 2 - 1
    j, k, l = NGWP_jkl(GP_dual, dr, dc)
    plot!(w, lw = 1, c = :black, legend = false, frame = :none, subplot = i,
          ylim = [minimum(w) * 1.5, maximum(w) * 1.5])
    # title = "ψ^($(j))_{$(k), $(l)}", titlefontsize = 6,
end
current()

## top HGLET
important_idx = sortperm(dvec_hglet[:].^2; rev = true)
plot(layout = Plots.grid(6, 8), size = (1200, 600))
# IB = zeros(N, 32)
for i in 1:48
    dr, dc = BS_hglet.levlist[important_idx[i]]
    w, _ = HGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                           GP, BS_hglet, G_Sig, method = :L)
    w .*= (maximum(w) > -minimum(w)) * 2 - 1
    # IB[:, i] = w
    j, k, l = HGLET_jkl(GP, dr, dc)
    plot!(w, lw = 1, c = :black, legend = false, frame = :none, subplot = i,
          ylim = [minimum(w) * 1.5, maximum(w) * 1.5])
end
current()
# wiggle(IB; sc = 0.75)

## top LP-HGLET
important_idx = sortperm(dvec_lphglet[:].^2; rev = true)
plot(layout = Plots.grid(6, 8), size = (1200, 600))
# IB = zeros(N, 32)
for i in 1:48
    dr, dc = BS_lphglet.levlist[important_idx[i]]
    w, _ = LPHGLET_Synthesis(reshape(spike(important_idx[i], N), (N, 1)),
                             GP, BS_lphglet, G_Sig; method = :L, ϵ = 0.3)
    w .*= (maximum(w) > -minimum(w)) * 2 - 1
    # IB[:, i] = w
    j, k, l = HGLET_jkl(GP, dr, dc)
    plot!(w, lw = 1, c = :black, legend = false, frame = :none, subplot = i,
          ylim = [minimum(w) * 1.5, maximum(w) * 1.5])
end
current()
# wiggle(IB; sc = 0.75)

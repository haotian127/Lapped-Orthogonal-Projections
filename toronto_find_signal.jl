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
@time LP_NGWP = lp_ngwp(𝚽, Gstar_Sig.W, GP_dual; ϵ = 0.4)

##
using Random
Random.seed!(1234)
μ, σ = [-79.4 43.71], 0.06
g = gen_gaussian(X, μ, σ)
P1, P2 = [-79.55 43.6], [-79.2 43.83]
g[line1side(X, P1, P2) .> 0] .-= 5
f = g + 1.5 * randn(N)
println(20 * log10(norm(f)/norm(f - g)))
# f .-= mean(f)
G_Sig.f = reshape(f, (N, 1))
gplot(A, X; width = 1);
signal_plt = scatter_gplot!(X; marker = f, plotOrder = :s2l, ms = 3)
savefig(signal_plt, "figs/Toronto_MG.png"); display(signal_plt)

##
plt = approx_curves_now(G_Sig, Gstar_Sig, GP_dual, VM_NGWP, LP_NGWP; ϵn = 0.4, ϵh = 0.05, frac = 0.5)
savefig(plt, "figs/Toronto_MG_approx.png"); display(plt)

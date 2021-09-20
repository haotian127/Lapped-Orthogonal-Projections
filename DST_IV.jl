cd(@__DIR__)

using MultiscaleGraphSignalTransforms, Plots, LightGraphs

N = 512; G = path_graph(N)
X = zeros(N,2); X[:, 1] = 1:N
L_dst4 = laplacian_matrix(G); L_dst4[1, 1] = 3
𝛌, 𝚽 = eigen(Matrix(L_dst4)); 𝚽 .*= sign.(𝚽[1, :])'
W = 1.0 * adjacency_matrix(G)

G_Sig = GraphSig(W, xy = X)
GP = partition_tree_fiedler(G_Sig; swapRegion = false)

# Uj = unitary_folding_operator(W, GP; ϵ = 0.3, J = 1)
fp

@time HGLET_dic = HGLET_DST4_dictionary(GP, G_Sig)
@time LPHGLET_dic = LPHGLET_DST4_dictionary(GP, G_Sig; ϵ = 0.3)

j = 3; k = 2; l = 6;
WH = HGLET_dic[GP.rs[k, j]:(GP.rs[k + 1, j] - 1), j, :]'
WlH = LPHGLET_dic[GP.rs[k, j]:(GP.rs[k + 1, j] - 1), j, :]'

plt = plot(WH[:, l], c = :black, grid = false, frame = :box, lw = 0.8,
     legendfontsize = 11, legend = false, size = (500, 400), ylim = [-0.15, 0.15])
     xticks!([1; 64:64:N], vcat(string(1), [string(k) for k in 64:64:N]))
# savefig(plt, "../figs/Path512_HGLET_j$(j-1)k$(k-1)l$(l-1).png")

plt = plot(WlH[:, l], c = :black, grid = false, frame = :box, lw = 0.8,
     legendfontsize = 11, legend = false, size = (500, 400), ylim = [-0.15, 0.15])
     xticks!([1; 64:64:N], vcat(string(1), [string(k) for k in 64:64:N]))
# savefig(plt, "../figs/Path512_LPHGLET_j$(j-1)k$(k-1)l$(l-1).png")

##
import WaveletsExt: wiggle

j = 4
for k = 1:8
   WH = HGLET_dic[GP.rs[k, j]:(GP.rs[k + 1, j] - 1), j, :]'
   plt = wiggle(WH[:, 1:8]; sc = 0.5)
         xticks!([1; 64:64:N], vcat(string(1), [string(k) for k in 64:64:N]))
   savefig(plt, "figs/Path512_HGLET_DST4_j$(j-1)_k$(k-1)_l(0_7).png")

   WlH = LPHGLET_dic[GP.rs[k, j]:(GP.rs[k + 1, j] - 1), j, :]'
   plt = wiggle(WlH[:, 1:8]; sc = 0.5)
         xticks!([1; 64:64:N], vcat(string(1), [string(k) for k in 64:64:N]))
   savefig(plt, "figs/Path512_LPHGLET_DST4_j$(j-1)_k$(k-1)_l(0_7).png")
end

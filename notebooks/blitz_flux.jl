### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# ╔═╡ 3a846134-a756-11eb-040c-83ac15be009f
begin
	using DrWatson
	quickactivate(findproject())
	
	using Pkg
	Pkg.instantiate()
	
	using PlutoUI
	using Flux
	using CUDA
end

# ╔═╡ 4a65c315-6e68-41fa-a898-a09e5ff7d93e
Pkg.add("Metalhead")

# ╔═╡ eb222d66-ff4e-4015-8bb7-ffd579445b87
using Flux: params

# ╔═╡ 4f86dd62-3c8e-4258-bc56-a8770439e0d9
with_terminal() do
	Pkg.status()
end

# ╔═╡ 78bb5b8f-fd9b-4e3e-a744-f8dda80ed9a3
TableOfContents()

# ╔═╡ 66be3273-a8c4-427c-89b9-a137475fca23
md"# Flux - A 60 Minute Blitz"

# ╔═╡ 1f17d0c7-11e2-4afa-a3e3-e4615390b3cb
md"## Arrays"

# ╔═╡ b0bf4962-d2ac-48bc-a430-fa1bcc9d7323
begin
	zeros(5, 5) .+ (1:5)', zeros(5, 5) .+ (1:5)
end

# ╔═╡ 9609fad0-1143-4777-800b-c95ed6159c58
(1:5) * (1:5)'

# ╔═╡ 6fc494ed-5c38-498d-967a-ede1220b724a
begin
	W = randn(5, 10)
	x = rand(10)
	W * x
end

# ╔═╡ 9ae68aad-43fc-4592-88e9-97f044863858
md"## CUDA Arrays"

# ╔═╡ d88d8f8f-e444-46ee-b7fe-ec729679df4b
cuArr = cu(rand(5, 3))

# ╔═╡ 257b9163-6509-4b78-87fc-1fdeaa975b98
md"## Automatic Differentiation"

# ╔═╡ 1a7df2ee-6bfb-4009-a581-d028d1d3fce4
f(x) = 3x^2 + 2x + 1

# ╔═╡ 54593e80-2e78-48dc-9190-f0d21aab3704
begin
	using Flux: gradient
	
	df(x) = gradient(f, x)[1]
	
	df(5)
end

# ╔═╡ bf33f485-c3f1-4c3d-82c3-5a1459762808
f(5)

# ╔═╡ f968322c-2889-4e7c-acd0-d046092094d6
ddf(x) = gradient(df, x)[1]

# ╔═╡ eb2ba24c-bd6c-4d40-80c6-083c1dacda03
df

# ╔═╡ 6f88ebf1-7fd2-44e6-9a9a-910b97f2fdc2
ddf(5)

# ╔═╡ 3e024376-b5bf-4f7b-b64e-a9ccf604fd41
mysin(x) = sum( (-1)^k * x^(1+2k) / factorial(1+2k) for k in 0:5)

# ╔═╡ 4954f848-2c9e-4682-8cfe-24a4c5e11026
xx = 0.5

# ╔═╡ 894e2d31-6f6e-40b7-b7ce-8fa42b8f25f3
mysin(xx), gradient(mysin, xx)

# ╔═╡ 24d86154-c442-47a9-813d-d35a7bbae028
sin(xx), cos(xx)

# ╔═╡ ceac59a0-5e67-405d-9b8a-e74361d6edd1
myloss(W, b, x) = sum(W * x .+ b)

# ╔═╡ 8c0eac6d-1c15-424e-b325-60cde19527ab
WW = randn(3, 5)

# ╔═╡ 0c967e9e-879f-4463-9d1b-2f2f2e5f901e
bb = zeros(3)

# ╔═╡ dcbb8e3a-a72b-4038-b3aa-1a4bba1d749c
xxx = rand(5)

# ╔═╡ d69ec190-3533-45c2-978e-3cbfe781bef0
gradient(myloss, WW, bb, xxx) 

# ╔═╡ 01513795-22c3-4b3e-97bb-ce6883477224
y(x) = sum( W * x .+ b)

# ╔═╡ 23be5a0c-19ab-4a4c-bca2-f9567f5b65ed
grads = gradient(()->y(x), params([W, b]))

# ╔═╡ 1499f882-cf9d-4c37-bef3-db2e19d8fef8
m = Dense(10, 5)

# ╔═╡ 31fef906-1837-4217-96fc-4e75b0916b9c
x2 = rand(Float32, 10)

# ╔═╡ c6b0535c-b816-4d55-a881-ab3eb2e63a99
params(m)

# ╔═╡ 4b65d8ee-3179-4850-90cc-d3c390111b7f
md"### compute gradients for all parameters"

# ╔═╡ 3be7dae3-3b95-4431-8811-3b4ec80bad06
begin
	x3 = rand(Float32, 10)
	mm = Chain(Dense(10, 5, relu), Dense(5, 2), softmax)
	l(x3) = sum(Flux.crossentropy(mm(x3), [0.5, 0.5]))
	grads2 = gradient(params(mm)) do
		l(x3)
	end
	for p in params(mm)
		println(grads2[p])
	end
end

# ╔═╡ 60f505ff-3f1f-46e1-bfed-e38e8802637b
begin
	using Flux.Optimise: update!, Descent
	η = 0.1
	for p in params(mm)
		update!(p, -η * grads2[p])
	end
end

# ╔═╡ a4eeed71-49d4-4fb0-84b0-5561e409b12e
params(mm)

# ╔═╡ 448c0a24-653d-4801-aa41-46144adde52a
md"### update weights and perform optimisation"

# ╔═╡ aae521c2-203e-48c1-ae6f-0c4353cde55e
opt = Descent(0.01)

# ╔═╡ 06f67eb7-2fab-4978-816a-25d92b53efc7
data, labels = rand(10, 100), fill(0.5, 2, 100)

# ╔═╡ 1c9262a2-7d3d-441a-ae14-5a04d8b47fd4
loss(x, y) = sum(Flux.crossentropy(mm(x), y))

# ╔═╡ 4698906f-972a-4524-b9d2-91a992fe3488
Flux.train!(loss, params(mm),  [(data, labels)], opt)

# ╔═╡ 80a6a3be-01b6-41ce-84d7-2d3e97ec720f
md"## Training a Classifier"

# ╔═╡ 747e358b-51e3-445b-8b17-c2998447efb4
md"Getting a real classifier to work might help cement the workflow a bit more. [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) is a dataset of 50k tiny training images split into 10 classes.

- Load CIFAR10 training and test datasets
- Define a Convolution Neural Network
- Define a loss function
- Train the network on the training data
- Test the network on the test data

"

# ╔═╡ f28cb5a3-e7fb-465a-b7b9-1b752b258dff


# ╔═╡ Cell order:
# ╠═3a846134-a756-11eb-040c-83ac15be009f
# ╠═4f86dd62-3c8e-4258-bc56-a8770439e0d9
# ╟─78bb5b8f-fd9b-4e3e-a744-f8dda80ed9a3
# ╟─66be3273-a8c4-427c-89b9-a137475fca23
# ╟─1f17d0c7-11e2-4afa-a3e3-e4615390b3cb
# ╠═b0bf4962-d2ac-48bc-a430-fa1bcc9d7323
# ╠═9609fad0-1143-4777-800b-c95ed6159c58
# ╠═6fc494ed-5c38-498d-967a-ede1220b724a
# ╟─9ae68aad-43fc-4592-88e9-97f044863858
# ╠═d88d8f8f-e444-46ee-b7fe-ec729679df4b
# ╟─257b9163-6509-4b78-87fc-1fdeaa975b98
# ╠═1a7df2ee-6bfb-4009-a581-d028d1d3fce4
# ╠═bf33f485-c3f1-4c3d-82c3-5a1459762808
# ╠═54593e80-2e78-48dc-9190-f0d21aab3704
# ╠═f968322c-2889-4e7c-acd0-d046092094d6
# ╠═eb2ba24c-bd6c-4d40-80c6-083c1dacda03
# ╠═6f88ebf1-7fd2-44e6-9a9a-910b97f2fdc2
# ╠═3e024376-b5bf-4f7b-b64e-a9ccf604fd41
# ╠═4954f848-2c9e-4682-8cfe-24a4c5e11026
# ╠═894e2d31-6f6e-40b7-b7ce-8fa42b8f25f3
# ╠═24d86154-c442-47a9-813d-d35a7bbae028
# ╠═ceac59a0-5e67-405d-9b8a-e74361d6edd1
# ╠═8c0eac6d-1c15-424e-b325-60cde19527ab
# ╠═0c967e9e-879f-4463-9d1b-2f2f2e5f901e
# ╠═dcbb8e3a-a72b-4038-b3aa-1a4bba1d749c
# ╠═d69ec190-3533-45c2-978e-3cbfe781bef0
# ╠═eb222d66-ff4e-4015-8bb7-ffd579445b87
# ╠═01513795-22c3-4b3e-97bb-ce6883477224
# ╠═23be5a0c-19ab-4a4c-bca2-f9567f5b65ed
# ╠═1499f882-cf9d-4c37-bef3-db2e19d8fef8
# ╠═31fef906-1837-4217-96fc-4e75b0916b9c
# ╠═c6b0535c-b816-4d55-a881-ab3eb2e63a99
# ╟─4b65d8ee-3179-4850-90cc-d3c390111b7f
# ╠═3be7dae3-3b95-4431-8811-3b4ec80bad06
# ╠═a4eeed71-49d4-4fb0-84b0-5561e409b12e
# ╟─448c0a24-653d-4801-aa41-46144adde52a
# ╠═60f505ff-3f1f-46e1-bfed-e38e8802637b
# ╠═aae521c2-203e-48c1-ae6f-0c4353cde55e
# ╠═06f67eb7-2fab-4978-816a-25d92b53efc7
# ╠═1c9262a2-7d3d-441a-ae14-5a04d8b47fd4
# ╠═4698906f-972a-4524-b9d2-91a992fe3488
# ╟─80a6a3be-01b6-41ce-84d7-2d3e97ec720f
# ╟─747e358b-51e3-445b-8b17-c2998447efb4
# ╠═4a65c315-6e68-41fa-a898-a09e5ff7d93e
# ╠═f28cb5a3-e7fb-465a-b7b9-1b752b258dff

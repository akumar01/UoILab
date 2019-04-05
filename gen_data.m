function [X, y, betas] = gen_data(varargin)

% Generate predictors from a multivariate distribution
p = inputParser;
addParameter(p, 'n_features', 100)
addParameter(p, 'n_informative', 10)
addParameter(p, 'n_samples', 100)
addParameter(p, 'noise', 0)
addParameter(p, 'random_state', now)
parse(p, varargin{:})

n_features = p.Results.n_features;
n_informative = p.Results.n_informative;
n_samples = p.Results.n_samples;
noise = p.Results.noise;
seed = p.Results.random_state;

X = mvnrnd(zeros(1, n_features), eye(n_features), n_samples);

% Draw betas from a uniform distribution

betas = 10 * rand(n_features, 1);

% Apply sparsity
betas(randsample(n_features, n_features - n_informative)) = 0;

% Generate data
y = X * betas + normrnd(0, noise, n_samples, 1);


end
function [X, y] = gen_data()

% Generate predictors from a multivariate distribution

X = mvnrnd(zeros(1, 10), eye(10), 100);

% Draw betas from a uniform distribution

betas = 10 * rand(10, 1);

y = X * betas + normrnd(0, 0.05, 100, 1);


end
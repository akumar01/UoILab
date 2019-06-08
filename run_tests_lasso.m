function run_tests_lasso

% test_variable_selection()
% test_estimation_score_usage()
test_random_state()
test_uoi_lasso_toy()
test_reg_params()
test_intercept()

end

function test_variable_selection

[X, y, w] = gen_data();
lasso = UoI_Lasso();
lasso = lasso.fit(X, y);

true_coef = nonzeros(w);
fit_coef = nonzeros(lasso.coef_);
assert_approx_equal(true_coef, fit_coef);

end

function test_estimation_score_usage

methods = {'r2', 'AIC', 'AICc', 'BIC'};

% Change to use random state to generate data
[X, y, ~] = gen_data('n_features', 10, 'n_informative', 3);

scores = [];
for i = 1:length(methods)
    lasso = UoI_Lasso('est_score', methods{i});
    assert(strcmp(lasso.est_score, methods{i}))
    lasso = lasso.fit(X, y);
    score = max(max(lasso.scores_));
    scores(i) = score;
end    

assert(length(unique(scores)) == length(methods))

end

function test_random_state

[X, y, ~] = gen_data('n_features', 5, 'n_informative', 3,...
                            'random_state', 16, 'noise', 0.5);

% Same state
l1log_0 = UoI_Lasso('random_state', 13);
l1log_1 = UoI_Lasso('random_state', 13);
l1log_0 = l1log_0.fit(X, y);
l1log_1 = l1log_1.fit(X, y);

assert_approx_equal(l1log_0.coef_, l1log_1.coef_)

% Different state
l1log_1 = UoI_Lasso('random_state', 14);
l1log_1 = l1log_1.fit(X, y);

keyboard
assert(~assert_approx_equal(l1log_0.coef_, l1log_1.coef_))

% Different state, not set
l1log_0 = UoI_Lasso();
l1log_1 = UoI_Lasso();
l1log_0 = l1log_0.fit(X, y);
l1log_1 = l1log_1.fit(X, y);

assert(~assert_approx_equal(l1log_0.coef_, l1log_1.coef_))


end

function test_uoi_lasso_toy

% Test UoI Lasso on a toy example %
X = [-1, 2; 4, 1; 1, 3; 4, 3; 8, 11];
beta = [1, 4];
y = X * beta;

% Choose the selection frac to be slightly smaller to ensure
% that we get good test sets

lasso = UoI_Lasso('fit_intercept', false, 'selection_frac', 0.75,...
                    'estimation_frac', 0.75);

lasso = lasso.fit(X, y);

assert_approx_equal(lasso.coef_, beta)

end

function test_reg_params
% Tests whether get_reg_params works correctly for UoI Lasso
X = [-1, 2; 0, 1; 1, 3; 4, 3];
y = [7, 4, 13, 16];

% Calculate the regularization parameter manually
alpha_max = max(X' * y)/4;
alphas = [alpha_max, alpha_max/10];

% Calculate reg params using UoI_Lasso object
lasso = UoI_Lasso('n_lambdas', 2, 'normalize', false,...
                'fit_intercept', false, 'tol', 0.1);

reg_params = lasso.get_reg_params(X, y);

assert_approx_equal(alphas, reg_params)

end

function test_intercept

% Test that UoI Lasso properly calculates the intercept
% when centering the response variable

X = [-1, 2; 0, 1; 1, 3; 4, 3];
y = [8, 5, 14, 17];

lasso = UoI_Lasso('normalize', false, 'fit_intercept', true);

assert(lasso.intercept_ == mean(y) - mean(X, 1) * lasso.coef_)

end

% function test_lasso_selection_sweep
% 
% X = [-1, 2, 3; 4, 1, -7; 1, 3, 1; 4, 3, 12; 8, 11, 2];
% beta = [1, 4, 2];
% y = X * beta;
% 
% % Toy regularization
% reg_param_values = [1, 2];
% coefs = lasso(X, y, 'Alpha', 1, 'lambda', reg_params);
% 
% 
% end


function assert_approx_equal(x1, x2, tol)

if nargin < 3
    tol = 1e-7;
end

assert(all(abs(x1 - x2) <= tol))

end
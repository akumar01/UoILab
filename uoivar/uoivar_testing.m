lambda = 10.^linspace(-2, 1, 20);
nlam = 20;
B1 = 30;
B2 = 20;
D = 1;
%parpool(2);

data = csvread('simdata4016.csv', 0, 0); % read in data
coef_mx = csvread('coef_mx40.csv', 0, 0); % read in coefficient matrix
L = 7; % set block length L used in randomization
[N, p] = size(data); % dataset dimensions: N rows are samples; p columns are features


%% uoi estimation
% blocks for block bootstrap

blocks = zeros(L, p, N - L); % storage for blocks: L by p by (N - L) array
% first two dimensions follow the same format as the data; third dimension
% is block index, i.e., blocks(:, :, B) is Bth block (colon indicates all)

for j = 1:(N - L) % for each feasible starting row j
    blocks(:, :, j) = data(j:(j + L - 1), :); % set down L rows beginning with j as jth block
end
ind = (N - L); % possible indices to randomly sample
n_blocks = ceil(N/L); % number of blocks needed to obtain bootstrap sample of same size as data


% storage for supports
supports = cell(nlam, B1);

tic
parfor k = 1:B1
    % block bootstrap
    rand_index = randi(ind, n_blocks, 1); % random selection of blocks
    bdata = zeros(n_blocks*L, p); % storage for bootstrap sample
    for j = 1:n_blocks % for each block position j
        bdata(((j - 1)*L + 1):(j*L), :) = blocks(:, :, rand_index(j)); % store jth randomly selected block in position j
    end
    remainder = n_blocks*L - N; % difference between bootstrap sample length and data length
    offset = randi([0 remainder], 1); % random offset for trimming to same size as data
    bdata = bdata((1 + offset):(N + offset), :); % trim to match original data size
    
    
    Z_mx = zeros(N - D, D*p); % storage for first rearrangement of data
    for j = 1:(N - D) % starting at each row until D from the end
        Z_mx(j, :) = reshape(bdata(j:(j + D - 1), :)', 1, D*p); % store D adjacent rows side-by-side
    end
    Y_mx = bdata(D + 1:N, :); % all but first D times
    
    % second rearrangement -- matrix operations
    X = kron(Z_mx, speye(p)); % compute kronecker product of Z and p x p identity
    Y = reshape(Y_mx', [], 1); % vectorize Y' matrix by column stacking
        
    % lasso supports
    supports(:, k) = findsupport(X, Y, lambda);
end
b1t = toc;

% intersection step
int_support = supports(:, 1);
for value = 1:nlam
    for k = 2:B1
        int_support{value} = intersect(int_support{value}, supports{value, k});
    end
end
clear supports;

% eliminate empty supports
int_support = int_support(cellfun(@(x) size(x, 1), int_support) > 0, :);

% union step
bhatcoefs = cell(B2, 1);
whichbest = zeros(1, B2);
tic
parfor m = 1:B2
    % block bootstrap training series
    rand_index = randi(ind, n_blocks, 1); % random selection of blocks
    bdata = zeros(n_blocks*L, p); % storage for bootstrap sample
    for j = 1:n_blocks % for each block position j
        bdata(((j - 1)*L + 1):(j*L), :) = blocks(:, :, rand_index(j)); % store jth randomly selected block in position j
    end
    remainder = n_blocks*L - N; % difference between bootstrap sample length and data length
    offset = randi([0 remainder], 1); % random offset for trimming to same size as data
    bdata = bdata((1 + offset):(N + offset), :); % trim to match original data size
    
    
    Z_mx = zeros(N - D, D*p); % storage for first rearrangement of data
    for j = 1:(N - D) % starting at each row until D from the end
        Z_mx(j, :) = reshape(bdata(j:(j + D - 1), :)', 1, D*p); % store D adjacent rows side-by-side
    end
    Y_mx = bdata(D + 1:N, :); % all but first D times
    
    % second rearrangement -- matrix operations
    bx_test = kron(Z_mx, speye(p)); % compute kronecker product of Z and p x p identity
    by_test = reshape(Y_mx', [], 1); % vectorize Y' matrix by column stacking
    rand_index = randi(ind, n_blocks, 1); % random selection of blocks
    bdata = zeros(n_blocks*L, p); % storage for bootstrap sample
    for j = 1:n_blocks % for each block position j
        bdata(((j - 1)*L + 1):(j*L), :) = blocks(:, :, rand_index(j)); % store jth randomly selected block in position j
    end
    remainder = n_blocks*L - N; % difference between bootstrap sample length and data length
    offset = randi([0 remainder], 1); % random offset for trimming to same size as data
    bdata = bdata((1 + offset):(N + offset), :); % trim to match original data size
    
    
    Z_mx = zeros(N - D, D*p); % storage for first rearrangement of data
    for j = 1:(N - D) % starting at each row until D from the end
        Z_mx(j, :) = reshape(bdata(j:(j + D - 1), :)', 1, D*p); % store D adjacent rows side-by-side
    end
    Y_mx = bdata(D + 1:N, :); % all but first D times
    
    % second rearrangement -- matrix operations
    bx_train = kron(Z_mx, speye(p)); % compute kronecker product of Z and p x p identity
    by_train = reshape(Y_mx', [], 1); % vectorize Y' matrix by column stacking
    
    % storage for least squares estimates and MSPE
    mspe = zeros(size(int_support, 1), 1);
    coefs = cell(size(int_support, 1), 1);
    
    % fit OLS and compute test set MSE
    for k = 1:size(int_support, 1)
        lm = fitlm(bx_train(:, int_support{k}), by_train, 'Intercept', false);
        by_hat = predict(lm, bx_test(:, int_support{k}));
        mspe(k) = mean((by_test - by_hat).^2);
        coefs{k} = lm.Coefficients{:, 1};
    end
    
    [~, whichbest(m)] = min(mspe);
    bhatcoefs{m} = coefs{whichbest(m)};
    %clear value best mspe coefs bx_test by_test bx_train by_train;
    
end
b2t = toc;

betahats = zeros(p.^2, B2);
for m = 1:B2
    betahats(int_support{whichbest(m)}, m) = bhatcoefs{m};
end
clear bhatcoefs;

% output
betahat_uoi = mean(betahats, 2);

beta = reshape(coef_mx', [], 1);
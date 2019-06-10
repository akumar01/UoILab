% data format: each row is one observation of a p-dimensional vector, and
% consecutive rows are consecutive times

data = csvread('simdata16016.csv', 0, 0); % read in data
coef_mx = csvread('coef_mx160.csv', 0, 0); % read in coefficient matrix
L = 7; % set block length L used in randomization
[N, p] = size(data); % dataset dimensions: N rows are samples; p columns are features

%% randomization

% a block of length L is a chunk of L consecutive rows from the dataset;
% for the randomization, blocks starting from each row from 1 to (N - L)
% are created; the blocks are randomly sampled and set down in order.

blocks = zeros(L, p, N - L); % storage for blocks: L by p by (N - L) array
% first two dimensions follow the same format as the data; third dimension
% is block index, i.e., blocks(:, :, B) is Bth block (colon indicates all)

for j = 1:(N - L) % for each feasible starting row j
    blocks(:, :, j) = data(j:(j + L - 1), :); % set down L rows beginning with j as jth block
end


% one randomization
rng(53017); % set rng seed
ind = (N - L); % possible indices to randomly sample
n_blocks = ceil(N/L); % number of blocks needed to obtain bootstrap sample of same size as data
rand_index = randi(ind, n_blocks, 1); % random selection of blocks
bdata = zeros(n_blocks*L, p); % storage for bootstrap sample
for j = 1:n_blocks % for each block position j
    bdata(((j - 1)*L + 1):(j*L), :) = blocks(:, :, rand_index(j)); % store jth randomly selected block in position j
end
remainder = n_blocks*L - N; % difference between bootstrap sample length and data length
offset = randi([0 remainder], 1); % random offset for trimming to same size as data
bdata = bdata((1 + offset):(N + offset), :); % trim to match original data size


%% reformatting for regression
% two rearrangements here: first, matrices Y, Z are constructed based
% on the var order, where for order D each row of the Z matrix is D
% adjacent rows from original data set side-by-side, and Y is the original
% data from times D + 1 to N; then the regression format comes from
% vectorizing Y by column-stacking and computing a kronecker product
% between Z and the identity matrix

D = 1; % VAR order

% first rearrangement -- to form Y' = BZ'
Z_mx = zeros(N - D, D*p); % storage for first rearrangement of data
for j = 1:(N - D) % starting at each row until D from the end
Z_mx(j, :) = reshape(bdata(j:(j + D - 1), :)', 1, D*p); % store D adjacent rows side-by-side
end
Y_mx = bdata(D + 1:N, :); % all but first D times

% second rearrangement -- matrix operations
X = kron(Z_mx, speye(p)); % compute kronecker product of Z and p x p identity
Y = reshape(Y_mx', [], 1); % vectorize Y' matrix by column stacking

% then X, Y are regression-formatted data   

% uoi procedure using above randomization and rearrangement estimates
beta = reshape(coef_mx', [], 1);
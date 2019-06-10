function out = sim_vard(p, T, d)
% output is a cell array, with first element simulated data, second element
% coefficient matrix, third element gaussian noise covariance, last element
% indices of nonzero coefficients in vectorized coefficient matrix

% p is the dimension of the response vector;
% T is the length of the simulated series;
% d is the order

%% matrix construction

pctNZ = 1/p; %% percent nonzero entries in transition matrices
numNZ = round(p*p*pctNZ); %% number nonzero entries

A_mxs = cell(1, d);
for k = 1:d
% fix nonzero terms in coefficient matrix

u_smp = unifrnd(0, 1, numNZ, 1); %% uniform sample
tht = 1; %% rate parameter for coefficient distribution
u_smp_trns = -log(1 - u_smp*(1 - exp(-8*tht))); %% inverse cdf transform uniform sample
NZcoef = (2*binornd(1, 0.5, numNZ, 1) - 1).*(8 - u_smp_trns); %% randomly choose sign

% randomly allocate nonzero coefficients to matrix

NZix = randi(p^2, numNZ, 1); %% indices
A_vec = zeros(p^2, 1); %% storage vector
A_vec(NZix) = NZcoef; %% assign
A_mx = reshape(A_vec, p, p); %% arrange as matrix


clear u_smp u_smp_trns

A_mxs{k} = A_mx;
end

long_A = A_mxs{1};
for k = 2:d
    long_A = [long_A A_mxs{k}];
end

% recondition matrix for stability

big_A = [long_A; eye((d - 1)*p) zeros((d - 1)*p, p)];
A_eigV = eig(big_A); %% eigenvalues
coef_mx = big_A/(max(abs(A_eigV))*2); %% coefficient matrix

long_coef_mx = coef_mx(1:p, :);

for k = 1:d
A_mxs{k} = long_coef_mx(:, ((k-1)*p + 1):(k*p));
end

%% simulate data recursively

n_initialize = 500; %% length of burn in period
N = n_initialize + T; %% total length

% construct block-diagonal covariance matrix

cov_parm = 0.7; %% value for off-diagonal covariances
blk_sz = 3; %% block size
nblk = floor(p/blk_sz); %% number of blocks
lastblk_sz = p - nblk*blk_sz; %% 'remainder' block
blk = (1 - 0.7)*eye(blk_sz) + 0.7; %% make block
cov_mx = blkdiag(kron(eye(nblk), blk), (1 - 0.7)*eye(lastblk_sz) + 0.7); %% put them together

% simulate Gaussian innovations

innov_mx = mvnrnd(zeros(1, p), cov_mx, N);

% recursively construct series
sim_series = zeros(N, p);
sim_series(1:d, :) = innov_mx(1:d, :);
for j = (d + 1):N
    ar_lags = zeros(d, p);
    for k = 1:d
        ar_lags(k, :) = sim_series(j - k, :)*(A_mxs{k}');
    end
    sim_series(j, :) = sum(ar_lags, 1) + innov_mx(j, :);
end


% subset by removing burn in

data_mx = sim_series((n_initialize + 1):N, :);

out = {data_mx, A_mxs, cov_mx, NZix, data_mx(T, :)};
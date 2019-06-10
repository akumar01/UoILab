function out = continue_var(T, A_mx, cov_mx, init_val)

d = length(A_mx); %% var order
p = size(A_mx{1}, 1);

%% simulate data recursively

N = T + 1; %% total length

% simulate Gaussian innovations

innov_mx = mvnrnd(zeros(1, p), cov_mx, N);

% recursively construct series

sim_series = zeros(N, p);
sim_series(1, :) = init_val;
for j = 2:N
    ar_lags = zeros(d, p);
    for k = 1:d
        ar_lags(k, :) = sim_series(j - k, :)*(A_mx{k}');
    end
    sim_series(j, :) = sum(ar_lags, 1) + innov_mx(j, :);
end

% subset by removing burn in

data_mx = sim_series(2:N, :);

out = {data_mx, data_mx(T, :)};

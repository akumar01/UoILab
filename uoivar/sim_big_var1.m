%% this file generates VAR(1) realization in pieces
% output is hdf5 file with simulated data, coefficients, and noise covariance
% references sim_var.m and continue_var.m -- these need to be in the
% working directory


T = 100000; %% chunk size
p = 354; %% dimension (think size of MEA array)
n_chunks = 5; %% number of chunks

N = n_chunks*T; %% total length of series to simulate

outpath = 'testhdf.h5'; %% output file path
h5create(outpath, '/sim_data', [Inf p], 'ChunkSize', [T p]); %% create file & dataset
count = [T p]; %% blocking of data (for writing chunkwise)

for chunk_num = 1:n_chunks
start = [((chunk_num - 1)*T + 1) 1]; %% location in h5 file to start writing

if chunk_num == 1 %% treat the first chunk differntly (generate coefficients, etc.)
chunk = sim_var(p, T);
A_mx = {chunk{2}};
cov_mx = chunk{3};
init_val = chunk{5};

data = chunk{1};
else %% for subsequent chunks generate the series as a continuation
chunk = continue_var(T, A_mx, cov_mx, init_val);
init_val = chunk{2};    

data = chunk{1};
end

h5write(outpath, '/sim_data', data, start, count); %% write chunk

max(init_val) %% print this to monitor whether any component series is diverging; shouldn't get too big

end

% store coefficient matrix
h5create(outpath, '/coef_mx', [p p]); 
h5write(outpath, '/coef_mx', A_mx{1});

% store noise covariance matrix
h5create(outpath, '/cov_mx', [p p]);
h5write(outpath, '/cov_mx', cov_mx);



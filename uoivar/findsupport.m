function supports = findsupport(x, y, lambda)
[B, ~] = lasso(x, y, 'Lambda', lambda);
[psq, nlam] = size(B);
ind = (1:psq)';
supports = cell(nlam, 1);
for i = 1:nlam
    supports{i} = ind(B(:, i) ~= 0);
end;
function K = gauss_K(X, Y, m, var, offset, scale)
% X = mesh of x coordinates
% Y = mesh of y coordinates
% m = mean of gaussian [m_x, m_y]
% var = covariance matrix [var_x, cov_xy; cov_xy, var_y]
% offset = prob value offset
% scale = scale the pdf values 
K = scale*mvnpdf([X(:), Y(:)], m, var) + offset;
K = reshape(K, size(X));
end
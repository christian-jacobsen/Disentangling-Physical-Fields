function K = circle_K(X, Y, c, r, k1, k2)
% X is matrix of x locations
% Y is matrix of y locations
% c is center of circle [x_c, y_c]
% r is radius of circle
% k1 is permeability outside circle
% k2 is permeability inside circle

K = k1*ones(size(X));
D = sqrt((X-c(1)).^2 + (Y-c(2)).^2);
K(D <= r) = k2;
end
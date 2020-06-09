function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

temp = z;
temp = -temp;
temp = exp(temp); %e^-z
temp = 1+temp;
temp = 1./temp;
g = temp;



% =============================================================

end

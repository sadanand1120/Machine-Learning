function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

temp_theta = theta;
temp_theta(1) = 0;

h = X*theta;
temp = h-y;
temp = temp.^2;
J = sum(temp);
temp = sum(temp_theta.^2);
temp = lambda*temp;
temp = J+temp;
J = temp/(2*m);


temp = h-y;
temp = X'*temp;
temp = temp+lambda*temp_theta;
grad = temp/m;



% =========================================================================

grad = grad(:);

end

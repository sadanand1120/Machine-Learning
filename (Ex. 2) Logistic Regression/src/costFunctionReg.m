function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


temp = X*theta;
h = sigmoid(temp); %hypothesis vector
t1 = log(h);
t1 = y.*t1;
t2 = log(1-h);
t2 = (1-y).*t2;
t = t1+t2;
t = sum(t);
t = -t/m;

temp = theta.^2;
temp = sum(temp)-theta(1)^2;
temp = temp/(2*m);
temp = temp*lambda;
J = t+temp;

%............................

temp = h-y;
temp = X'*temp;
stor = temp(1);
temp = temp + lambda*theta;
temp(1) = stor;
grad = temp/m;




% =============================================================

end

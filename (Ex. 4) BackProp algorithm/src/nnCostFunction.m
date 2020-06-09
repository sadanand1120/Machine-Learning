function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(m,1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%......................................................

%%%%%% WORKING WITH FOR LOOP
%%y_local = zeros(num_labels,m);  % num_labels x m
%%for i=1:m,
%%	y_local(y(i),i)=1;
%%	end;

%%%%% Implementation vectorized without for loop
temp = (1:num_labels)==y;  % m x num_labels
y_local = temp';

% X =========> m x (input_size+1)
% Theta1 ====> h_size x (input_size+1)
% Theta2 ====> num_labels x (h_size+1)

temp = Theta1*X';  % h_size x m
temp = sigmoid(temp);
temp = [ones(1,m); temp];   % (h_size+1) x m
temp = Theta2*temp;  % num_labels x m
h = sigmoid(temp);


%cost calculation
temp = (y_local.*log(h))+[(1-y_local).*log(1-h)];
temp = sum(sum(temp));
temp = -temp/m;
J = temp;

%regularization
theta1 = Theta1(:,2:end);
theta2 = Theta2(:,2:end);
theta1 = theta1.^2;
theta2 = theta2.^2;
temp = sum(sum(theta1)) + sum(sum(theta2));
temp = (lambda*temp)/(2*m);
J = J+temp;

%........................................................

% X =========> m x (input_size+1)
% Theta1 ====> h_size x (input_size+1)
% Theta2 ====> num_labels x (h_size+1)

triangle1 = zeros(size(Theta1));
triangle2 = zeros(size(Theta2));

%%%%%%%%%%%%% USING FOR LOOP GRADIENT CALCULATION
%%%%%%%%%%%%% (slow)
%for t=1:m,
%	%Forward Propagation
%	a1 = (X(t,:))';  % (input_size+1) x 1
%	temp = Theta1*a1;  % h_size x 1
%	temp = sigmoid(temp);
%	a2 = [1; temp];   % (h_size+1) x 1
%	temp = Theta2*a2;  
%	a3 = sigmoid(temp);  % num_labels x 1
%
%	delta3 = a3-y_local(:,t);   % num_labels x 1
%	%BackPropagation
%	temp = Theta2'*delta3;   %(h_size+1) x 1
%	delta2 = temp.*a2.*(1-a2);
%	
%	delta2 = delta2(2:end);  % h_size x 1
%	triangle1 = triangle1 + delta2*a1';
%	triangle2 = triangle2 + delta3*a2';	
%end;
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%% VECTORIZED IMPLEMENTATION
%%%%%%%%%%%%
% ForwardProp
a1 = X';            % (input_size+1) x m
temp = Theta1*a1;   % h_size x m
temp = sigmoid(temp);
a2 = [ones(1,m); temp]; % (h_size+1) x m
temp = Theta2*a2;
a3 = sigmoid(temp); % num_labels x m

delta3 = a3-y_local;
% BackProp
temp = Theta2'*delta3;  % (h_size+1) x m
delta2 = temp.*a2.*(1-a2);

delta2 = delta2(2:end,:);  % h_size x m
triangle1 = triangle1 + delta2*a1'; % automatically sum over m 
triangle2 = triangle2 + delta3*a2'; % automatically sum over m
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%

Theta1_grad = (triangle1/m) + (lambda/m)*Theta1;
Theta2_grad = (triangle2/m) + (lambda/m)*Theta2;
Theta1_grad(:,1) = Theta1_grad(:,1) - (lambda/m)*Theta1(:,1);
Theta2_grad(:,1) = Theta2_grad(:,1) - (lambda/m)*Theta2(:,1);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

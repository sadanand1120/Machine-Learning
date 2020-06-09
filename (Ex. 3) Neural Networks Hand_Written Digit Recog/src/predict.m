function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
n = size(X, 2);
num_labels = size(Theta2, 1);
h_size = size(Theta1, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m,1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


% X ============> m x (n+1)
% theta1 =======> h_size x (n+1)
% theta2 =======> num_labels x (h_size+1)

temp = Theta1*X';  % h_size x m
temp = sigmoid(temp);
temp = [ones(1,m); temp]; % (h_size+1) x m
temp = Theta2*temp;  % num_labels x m
temp = sigmoid(temp);

[val ind] = max(temp,[],1);
p = ind';




% =========================================================================


end

function p = predict(Theta1, Theta2, X)
% p: prediction
% Theta1: Parameters of the regresion.
% Theta2: Parameters of the regresion.
% X: Training examples of the data whithout feature y.


m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);


a2 = sigmoid([ones(m, 1) X] * Theta1');
a3 = sigmoid([ones(m, 1) a2] * Theta2');
[dummy, p] = max(a3, [], 2);

end

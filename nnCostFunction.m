function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% J: Cost of the regresion with theta.
% grad: Gradient of J.
% nn_params: Parameters of the neural network.
% input_layer_size: Nº de características de entrada.
% hidden_layer_size: Nº de características de la capa oculta.
% num_labels: Nº de decisiones posibles.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% lambda: Paramer of the regularization.





Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
A1 = [ones(m, 1) X];  
num_labels = size(Theta2, 1);
J = 0;
p = zeros(size(X, 1), 1);  
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% Real values of decision (as matrix of 0,1).
yv=zeros(size(X,1),num_labels);
yv=[1:num_labels]==y;


% Forward propagation
Z2 = A1*Theta1';
A2 = [ones(m, 1) sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = [ones(m, 1) sigmoid(Z3)];


% Error of predictions
Err=sum(sum(yv.*log(A3(:,2:end))+(1-yv).*log(1-A3(:,2:end))));
reg=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J=-(1/m)*Err+reg;


% Back propagation
delta3 = (A3(:,2:end)-yv);
delta2 = (delta3*Theta2(:,2:end)).*sigmoidGradient(Z2);
Delta1 = A1'*delta2;
Delta2 = A2'*delta3;
Theta1_grad = (1/m)*Delta1';
Theta2_grad = (1/m)*Delta2';
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+ (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+ (lambda/m)*Theta2(:,2:end); 
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function checkNNGradients(lambda,Theta1,Theta2,X,y, ...
				    input_layer_size, ...
				    hidden_layer_size, ...
				    num_labels)
% lambda: Paramer of the regularization.
% Theta1: Parameters of the neural network.
% Theta2: Parameters of the neural network.
% X: Training examples of the data whithout feature y.
% y: Training examples of the feature y.
% input_layer_size: Nº de características de entrada.
% hidden_layer_size: Nº de características de la capa oculta.
% num_labels: Nº de decisiones posibles.



% Initial values
if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end
nn_params = [Theta1(:) ; Theta2(:)];
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);


% Compute gradient of first iteration.
[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

fprintf(['(Left-Numerical Gradient, Right-Analytical Gradient)\n\n']);
disp([numgrad(1:10) grad(1:10)]);
fprintf(['\nThe above two columns you get should be very similar.\n']);


% Relative diference.
diff = norm(numgrad-grad)/norm(numgrad+grad);
fprintf(['The relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n\n\n'], diff);

end







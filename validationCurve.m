function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, 									initial_nn_params,options);
% lambda_vec: values of lambda.
% error_train: error of train set.
% error_val: error of cross validation set.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% input_layer_size: number of input features.
% hidden_layer_size
% num_lables: number of labels.
% initial_nn_params: initial parameters.
% options: options of the cost funstion minimization.


             

lambda_vec = [0 0.001 0.003 0.03 0.3 3 10]';
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
         
	lambda = lambda_vec(i,1);
     fprintf('\n Case lambda= %f\n',lambda);
	costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
     [nn_params, cost] = fmincg(costFunction, 	initial_nn_params, options);


	error_train(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, 0);


	error_val(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xval, yval, 0);



endfor


end

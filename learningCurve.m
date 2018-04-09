function [error_train, error_val,ind] = ...
    learningCurve(X, y, Xval, yval, lambda, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, initial_nn_params,options)
% error_train: error of train set.
% error_val: error of cross validation set.
% ind: index of number of examples.
% X: X train set.
% y: y train set.
% Xval: X cross validation set.
% yval: y cross validation set.
% lambda: parameter of regularization.
% input_layer_size: number of input features.
% hidden_layer_size: number of hidden features.
% num_labels: number of labels.
% initial_nn_params: initial values of theta.
% options: options to minimize cost function


m = size(X, 1);
num=15;
error_train = zeros(num+1, 1);
error_val   = zeros(num+1, 1);
valu=floor(m/num)-2;
ind=0;


for i=1:num+1,
	
	value=(i-1)*valu+2;
	ind=[ind value];
     fprintf('\n Case m= %f\n',ind(i+1));
	x_train = X(1:value,:);
	y_train = y(1:value);	
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, x_train, y_train, lambda);
[nn_params, cost] = fmincg(costFunction, 	initial_nn_params, options);

     error_train(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, x_train, y_train, 0);


	error_val(i) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xval, yval, 0);

endfor
   
ind=ind(2:num+2);

error_train
error_val



end







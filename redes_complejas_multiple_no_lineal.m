
%% ................................................
%% ................................................
%%       REDES COMPLEJAS MULTIPLE NO LINEAL
%% ................................................
%% ................................................






%% 1. Clear and Close Figures
clear ; close all; clc





%% ==================== Part 1: Data ====================
fprintf('\n \nDATA\n.... \n \n \n');   





%% 2. Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add your own file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


fprintf('Loading data ...\n'); 
%%%%%%********Select features********   
input_layer_size  = 400;  
hidden_layer_size = 20;   
num_labels = 10;  
%%%%%%********Select archive********   
load('ex4data1.mat');        
m = size(X, 1);
fprintf('(X,y) (10 first items and 5 first columns)\n');   
[X(1:10,1:5) y(1:10,:)]
fprintf('Program paused. Press enter to continue.\n\n\n\n');
pause;


%% 4. Select train, cross and test validation sets
[X, y, Xval, yval, Xerr, yerr, m, n] = ...
    selectsets(X, y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%% ========= Part 2: Checking Backpropagation ================
fprintf('CHECKING BACKPROPAGATION\n........................ \n \n \n \n');





%% 5. Initial values
%%%%% ****Select the number of debug values and lambda ******
num_rand=10;
lambda = 1;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% 6. Checking backpropagation
more off;
fprintf('Checking Backpropagation... (it can last a few minits)\n\n')
sel = randperm(size(X, 1));
sel = sel(1:num_rand);
X_check=X(sel,:);
y_check=y(sel,:);
checkNNGradients(lambda,initial_Theta1,initial_Theta2, ...
			 X_check, y_check, ...
	            input_layer_size, ...
		       hidden_layer_size, ...
		       num_labels);
fprintf('Program paused. Press enter to continue.\n\n\n\n');
pause;





%% =================== Part 3: Training NN ===================
fprintf('TRAINING NEURAL NETWORK\n........................ \n \n \n \n');





%% 6. Neural network
%%%%% *************Select iterations***********
num_iters=50;
options = optimset('MaxIter', num_iters);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% extract theta

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional: Execute your own gradient descent.

%fprintf('Running gradient descent with alpha ... \n \n ');
%%%%% *************Select iterations***********
%num_iters = 1000;
%[theta, J_his] = gradientDescentMulti(X, y, theta, alpha, %num_iters);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 7. Display results
fprintf('\n \nTheta1 (first 5 rows and colums): \n');
Theta1(1:5,1:5)
fprintf('\n');
fprintf('\n \nTheta2 (first 5 rows and colums): \n');
Theta2(1:5,1:5)
fprintf('\n');
pred = predict(Theta1, Theta2, X);
pred1 = predict(Theta1, Theta2, Xval);
pred2 = predict(Theta1, Theta2, Xerr);

error = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, 0);

error_c = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xval, yval, 0);

error_t = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, Xerr, yerr, 0);

fprintf('\nTraining Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nCross Accuracy: %f\n', mean(double(pred1 == yval)) * 100);
fprintf('\nTest Accuracy: %f\n', mean(double(pred2 == yerr)) * 100);
fprintf('\nTraining Error: %f\n', error);
fprintf('\nCross Error: %f\n', error_c);
fprintf('\nTest Error: %f\n', error_t);
fprintf('\nProgram paused. Press enter to continue.\n \n \n \n');
pause;





%% ============== Part 3: Sample to predict  ==============
fprintf('SAMPLE\n...... \n \n \n \n');





%% 8. Select a sample to predict
%%%%% *************Select sample to predict***********
x11=X(80,:);
estimation_y = predict(Theta1,Theta2, x11);
fprintf('Prediction:\n x= \n');
fprintf('%f  \n',x11(1, 2:8));
fprintf('...\n');
fprintf('\n y_pred= %f',estimation_y);
fprintf('\n y_real= %f \n \n',y(80));

fprintf('Program paused. Press enter to continue.\n');
pause;





%% ==== Part 11: Learning Curve for Linear Regression ========
fprintf('\n\n LEARNING CURVE\n............... \n \n \n \n');





[error_train, error_val, ind] = ...
    learningCurve(X, y, Xval, yval, lambda, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, 									initial_nn_params,options);

plot(ind, error_train, ind, error_val);
title('Learning curve')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
fprintf('Check if there is a bios or variation problem.\n\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;





%% ================ Part 12: Validation ================
fprintf('\n\nVALIDATION\n........... \n \n \n \n');





[lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, 									initial_nn_params,options);


figure;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
fprintf('\nlambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end


fprintf('\n Actual lambda: \n');
fprintf(' %f \n', lambda);
fprintf('\nThe best lambda has the lowest validation error.\n\n');
fprintf('Program paused. Press enter to continue.\n');
pause;























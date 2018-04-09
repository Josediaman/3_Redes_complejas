function g = sigmoidGradient(z)
% g: value of the derivate of sigmoid function in z.
% z: value/set of values. 

g = sigmoid(z).*(1-sigmoid(z));

end

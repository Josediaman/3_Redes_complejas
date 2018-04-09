function W = randInitializeWeights(L_in, L_out)
% W: Random Values.
% L_in: Input layer size.
% L_out: Output layer size. 



W = zeros(L_out, 1 + L_in);
eps = sqrt(6)/sqrt(L_in+L_out);
W = rand(L_out,L_in+1)*2*eps-eps;

end

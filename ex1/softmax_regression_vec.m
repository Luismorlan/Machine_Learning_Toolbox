function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
%   theta_full = [theta zeros(n,1)];
%   theta_k = theta_full(:,y);
% %%%convert
%   keys = eye(10);
%   bool = keys(y,:);
%   bool = bool(:,1:9);
%   dev = (sum(exp(theta'*X))'*ones(1,9));
%   f = -sum(log(exp(sum(theta_k.*X))./exp(sum(theta'*X))));
%   g = -X*(bool - exp(X'*theta)./dev);
%   
  softmaxInput = [theta'*X;ones(1,m)]; %num_classes*num_examples
  softmaxOutput = bsxfun(@rdivide, exp(softmaxInput), sum(exp(softmaxInput)));
  I = sub2ind(size(softmaxOutput),y,1:m);
  f = -sum(log(softmaxOutput(I)));
  temp = zeros(size(softmaxOutput));
  temp(I) = 1;
  gradInput = temp-softmaxOutput;
  g = -X*gradInput';
  g = g(:,1:num_classes-1);
  
  g=g(:); % make gradient a vector for minFunc


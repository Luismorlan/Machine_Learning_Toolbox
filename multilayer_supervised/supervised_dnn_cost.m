function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
if numHidden>=1,
    hAct{1,1} = sigmoid(stack{1,1}.W*data+stack{1,1}.b*ones(1,size(data,2)));
    for i = 1:numHidden-1,
        hAct{i+1,1} = sigmoid(stack{i+1,1}.W*hAct{i,1}+stack{i+1,1}.b*ones(1,size(hAct{i,1},2)));
    end
    temp = (stack{numHidden+1,1}.W*hAct{numHidden,1}+stack{numHidden+1,1}.b*ones(1,size(hAct{numHidden,1},2)));
    temp_1 = exp(temp);
    hAct{numHidden+1,1} = temp_1./(ones(size(temp_1,1),1)*sum(temp_1));
else,
    temp = (stack{1,1}.W*data+stack{1,1}.b*ones(1,size(data,2)));
    temp_1 = exp(temp);
    hAct{1,1} = temp_1./(ones(size(temp_1,1),1)*sum(temp_1));
end
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
I = sub2ind(size(hAct{numHidden+1,1}),labels', 1:size(hAct{numHidden+1,1},2));
cost = -sum(log(hAct{numHidden,1}(I)));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
conv = eye(ei.output_dim);
bool = conv(:,labels);

delta = (bool+hAct{numHidden+1,1});
last_layer_grad_W = delta*hAct{numHidden,1}';
last_layer_grad_b = delta*ones(size(data,2),1);

gradStack{numHidden+1,1}.W = last_layer_grad_W;
gradStack{numHidden+1,1}.b = last_layer_grad_b;

for i = numHidden:-1:2,
    delta = stack{i+1,1}.W'*delta;
%    gradStack{i,1}.W = delta*hAct
end

a = stack{2,1}.W'*delta;
delta = a.*(1-a);
gradStack{1,1}.W = delta*data';
gradStack{1,1}.b = delta*ones(size(data,2),1);



%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end




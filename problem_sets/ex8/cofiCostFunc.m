function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


H = X*Theta';
HY = (H-Y);
HY2 = HY.^2;
J = sum(sum(HY2.*R))/2;



% for i=1:size(X,1)
%     for k=1:size(X,2)
%         s = 0;
%         for j=1:size(Theta,1)
%             if(R(i,j) == 1)
%                 s = s + (Theta(j,:)*X(i,:)'-Y(i,j))*Theta(j,k);
%             end;
%         end;
%         X_grad(i,k) = s;
%     end;
% end;

for i=1:size(X,1)
    idx = find(R(i,:) == 1);
    ThetaTemp = Theta(idx,:);
    YTemp = Y(i,idx);
    X_grad(i,:) = (X(i,:)*ThetaTemp'-YTemp)*ThetaTemp + lambda.*X(i,:);
end

for j=1:size(Theta,1)
    idx = find(R(:,j) == 1);
    XTemp = X(idx,:);
    YTemp = Y(idx,j);
    Theta_grad(j,:) = (XTemp'*(XTemp*Theta(j,:)'-YTemp))' + lambda.*Theta(j,:);
end;


% for j=1:size(Theta,1)
%     for k=1:size(Theta,2)
%         s = 0;
%         for i=1:size(X,1)
%             if(R(i,j) == 1)
%                 s = s + (Theta(j,:)*X(i,:)'-Y(i,j))*X(i,k);
%             end;
%         end;
%         Theta_grad(j,k) = s;
%     end;
% end;


J = J + sum(sum((Theta.^2)))*lambda/2 + sum(sum((X.^2)))*lambda/2;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

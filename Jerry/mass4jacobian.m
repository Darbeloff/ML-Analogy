function [A,B] = mass4jacobian(x,u, params)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
A= zeros(length(x),length(x));
A(1,2) = 1;
A(3,4) = 1;
A(5,6) = 1;
A(7,8) = 1;

A(2,1) = -1 - 3*(x(1) - x(3))^2;
A(2,3) = 1 + 3*(x(1) - x(3))^2;
A(2,2) = -2*(x(2)-x(4));
A(2,4) = 2*(x(2)-x(4));

A(4,1) = 1 + 3*(x(1) - x(3))^2;
A(4,3) = -1 - 3*(x(1) - x(3))^2 - 1 - 3*(x(3) - x(5))^2;
A(4,5) = 1 + 3*(x(3) - x(5))^2;
A(4,2) = 2*(x(2)-x(4));
A(4,4) = -2*(x(2)-x(4)) - 2*(x(4) - x(6));
A(4,6) = 2*(x(4)-x(6));

A(6,3) = 1 + 3*(x(3) - x(5))^2;
A(6,5) = -1 - 3*(x(3) - x(5))^2 - 1 - 3*(x(5) - x(7))^2;
A(6,7) = 1 + 3*(x(5) - x(7))^2;
A(6,4) = 2*(x(4)-x(6));
A(6,6) = -2*(x(4)-x(6)) - 2*(x(6) - x(8));
A(6,8) = 2*(x(6)-x(8));

A(8,5) = 1 + 3*(x(5) - x(7))^2;
A(8,7) = -1 - 3*(x(5) - x(7))^2;
A(8,6) = 2*(x(6)-x(8));
A(8,8) = -2*(x(6)-x(8));
fk1 = -(x(1) - x(3)) - (x(1) - x(3))^3;
fd1 = -(x(2) - x(4))^2 *sign(x(2) - x(4));
fk2 = -(x(3) - x(5)) - (x(3) - x(5))^3;
fd2 = -(x(4) - x(6))^2 *sign(x(4) - x(6));
fk3 = -(x(5) - x(7)) -(x(5) - x(7))^3;
fd3 = -(x(6) - x(8))^2 *sign(x(6) - x(8));
B = zeros(length(x),length(u));
B(2) = 1;
end


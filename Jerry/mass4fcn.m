function [dxdt] = mass4fcn(x,u, params)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
dxdt = zeros(size(x));
fk1 = -(x(1) - x(3)) - (x(1) - x(3))^3;
fd1 = -(x(2) - x(4))^2 *sign(x(2) - x(4));
fk2 = -(x(3) - x(5)) - (x(3) - x(5))^3;
fd2 = -(x(4) - x(6))^2 *sign(x(4) - x(6));
fk3 = -(x(5) - x(7)) -(x(5) - x(7))^3;
fd3 = -(x(6) - x(8))^2 *sign(x(6) - x(8));
dxdt(1) = x(2);
dxdt(2) = fk1 + fd1 + u;
dxdt(3) = x(4);
dxdt(4) = - fk1 - fd1 + fk2 + fd2;
dxdt(5) = x(6);
dxdt(6) = - fk2 - fd2 + fk3 + fd3;
dxdt(7) = x(8);
dxdt(8) = - fk3 - fd3;
end


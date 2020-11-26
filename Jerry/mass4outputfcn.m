function [y] = mass4outputfcn(x,u, params)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
h = zeros(6,1);
h(1) = (x(1) - x(3)) + (x(1) - x(3))^3;
h(2) = (x(3) - x(5)) + (x(3) - x(5))^3;
h(3) = (x(5) - x(7)) + (x(5) - x(7))^3;
h(4) = (x(2) - x(4))^2*sign(x(2) - x(4));
h(5) = (x(4) - x(6))^2*sign(x(4) - x(6));
h(6) = (x(6) - x(8))^2*sign(x(6) - x(8));
y = [x];
end


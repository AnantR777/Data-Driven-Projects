a = 1;
b = 0.1;
c = 1;
f = @(t,x) [x(3)+(x(2)-a)*x(1);1-b*x(2)-x(1)^2;-x(1)-c*x(3)];
% we made x the input vector so x(1) = x, x(2) = y, x(3) = z
[t,xa] = ode45(f,[0 100],[3.75 3.5 2.98]); % solves system of diffeqs
% second argument is range, third is ICs
plot(t,xa)
title('x(t), y(t), z(t)')
xlabel('t'), ylabel('x,y,z')
legend('x(t) = interest rate','y(t) = investment demand','z(t) = price index')
% illustrates the relationship between the 3 parameters

%% helper function
function z = radian_distance(x,y) %mod 180
z = zeros(size(x));
for i = 1:length(x(:))
coordinatesA = [ cos(2*x(i)) sin(2*x(i))];
coordinatesB = [ cos(2*y(i)) sin(2*y(i))];
z(i) = real(acos(dot(coordinatesA',coordinatesB')'./(sqrt(sum(coordinatesA.*coordinatesA,2)).*sqrt(sum(coordinatesB.*coordinatesB,2)))))/2;
end
end
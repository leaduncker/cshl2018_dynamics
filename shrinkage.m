function y = shrinkage(a, kappa)
% shrinkage operator applied elementwise to a
    y = max(0, a-kappa) - max(0, -a-kappa);
end
function experiment = evaluate_detscore(experiment)

M = experiment.FImatrix;
invM = inv(M);

experiment.detscore = det(invM);


end
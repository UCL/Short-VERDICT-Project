%%%% Dice score calculation
function score=calculate_dice_score(grndTruth,segIm);
score = 2*nnz(segIm&grndTruth)/(nnz(segIm) + nnz(grndTruth));
end
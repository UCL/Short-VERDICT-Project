
newstruct = scores;

newstruct = rmfield(newstruct, 'PatNums');
newstruct = rmfield(newstruct, 'scores');
newstruct = rmfield(newstruct, 'schemename');
newstruct = rmfield(newstruct, 'fittingtechnique');
newstruct = rmfield(newstruct, 'modeltype');
newstruct = rmfield(newstruct, 'biopsyresults');
newstruct = rmfield(newstruct, 'sensitivity');
newstruct = rmfield(newstruct, 'specificity');
newstruct = rmfield(newstruct, 'thresholds');

table = struct2table(newstruct);

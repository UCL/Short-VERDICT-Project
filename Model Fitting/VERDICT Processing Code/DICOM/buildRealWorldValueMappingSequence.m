function RealWorldValueMappingSequence = buildRealWorldValueMappingSequence(last,first,slope,intercept)
%
%
% dinfo = addfield(dinfo,RWVMS) ;

% Experimental

% MeasurementUnitsCodeSeqeunce
CodeValue = '1' ; % ?
CodingSchemeDesignator = 'UCUM' ;  %  Unified Code for Units of Measure
CodeMeaning = 'xxxxx' ; % ?
ContextUID = dicomuid ; % ?

LUTExplanation = 'Volume fraction' ;
LUTLabel = 'Volume fraction' ;
RealWorldValueLastValueMapped = last ;
RealWorldValueFirstValueMapped = first ;
RealWorldValueIntercept  = slope ;
RealWorldValueSlope      = intercept ;


RealWorldValueMappingSequence.Item_1.MeasurementUnitsCodeSequence.Item_1.CodeValue = CodeValue ;
RealWorldValueMappingSequence.Item_1.MeasurementUnitsCodeSequence.Item_1.CodingSchemeDesignator = CodingSchemeDesignator ;
RealWorldValueMappingSequence.Item_1.MeasurementUnitsCodeSequence.Item_1.CodeMeaning = CodeMeaning ;
RealWorldValueMappingSequence.Item_1.MeasurementUnitsCodeSequence.Item_1.ContextUID = ContextUID ;

RealWorldValueMappingSequence.Item_1.LUTExplanation = LUTExplanation ;
RealWorldValueMappingSequence.Item_1.LUTLabel = LUTLabel ;
RealWorldValueMappingSequence.Item_1.RealWorldValueLastValueMapped = RealWorldValueLastValueMapped ;
RealWorldValueMappingSequence.Item_1.RealWorldValueFirstValueMapped = RealWorldValueFirstValueMapped ;
RealWorldValueMappingSequence.Item_1.RealWorldValueIntercept = RealWorldValueIntercept ;
RealWorldValueMappingSequence.Item_1.RealWorldValueSlope = RealWorldValueSlope ;
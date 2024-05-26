%% Receiver Operating Characteristic (ROC) Curves
% ROC curves provide a way to assess a new diagnostic test. This applies to 
% tests that give a numercial *score* and from this score we want to classify 
% a patient as positive for disease or negative.The aim is often to characterise 
% a new test that we hope has some useful feature such as being cheaper, easier, 
% fewer side-effects etc. During this assessment phase, we need a reference standard 
% to give what we assume is a 'ground truth' *label*. Example scores could be 
% the intensity in an image region or the concentration of a compoud in a blood 
% sample. Example positive labels could be 'malignant' or 'sick' and negative 
% labels might be 'benign' or 'healthy'. If the new test is to be used, we will 
% need to choose a threshold score. We will classify all scores above this threshold 
% (or operating point) as positive and all below as negative. The ROC plot shows 
% how the test performs for different possible values of the threshold score. 
% The area under the curve (AUC) provides an overall measure of how good the test 
% is (the closer the AUC is to 1, the better).
% 
% The reason the AUC is less than 1 is related to the fundamental problem that 
% some scores for positive and negative patients overlap. The result is that for 
% most useful choices of a threshold, there will be some false negatives and some 
% false positives. This is perhaps best visualised first using a histogram.
% 
% 
% 
% Create an illustrative example from normal distributions with the mean score 
% being 10 for positives being and 5 for negatives. Intuitively, we expect the 
% threshold to be somewhere between 5 and 10. 

% Simualate scores for positive and negative patients
scores_pos = 10 + 3*randn([200 1]) ; % scores for positive (sick) patients
scores_neg =  5 + 3*randn([250 1]) ; % scores for negative (healthy) patients

scores_all = [scores_pos(:) ; scores_neg(:)] ;
labels_all = cat(1,repmat('Positive',[length(scores_pos) 1]), repmat('Negative',[length(scores_neg) 1])) ;

disp("Median scores: positive " + median(scores_pos) + ", negative "+median(scores_neg))


hf = figure;
tiledlayout(2,2)
% Visualise the scores using a histogram
axhist = nexttile([1 1]) ;
histogram(scores_pos,'Orientation','horizontal','FaceColor',[1 0 0])
hold on
histogram(scores_neg,'Orientation','horizontal','FaceColor',[0 0 1])
ylabel('Score'), xlabel('Frequency')
grid on, ylim([min(scores_all) max(scores_all)])

% Visualise using a boxplot (ignore warning)
axbox = nexttile([1 1]) ;
boxplot(scores_all, labels_all, 'Notch','on')
grid on, ylim([min(scores_all) max(scores_all)])


% ROC Curve
axroc = nexttile([1 2]) ;
[X,Y,T,AUC,OPTROCPT] = perfcurve(labels_all, scores_all,'Positive','NBoot',1000) ;
% Columns are 1 mean, 2 lower confidence, 3 upper confidence
hp = plot(X(:,1),Y(:,1),'LineWidth',2) ;
hold on
plot([0 1],[0 1],'LineWidth',1) % diagonal line
xlabel('False Positive Rate (1-Specificity)')
ylabel('True Positive Rate (Sensitivity)')
grid on

% Set data tip so that it also shows the thresholds at each point
dtt = hp.DataTipTemplate;
dtt.DataTipRows(1).Label = 'FPR' ;
dtt.DataTipRows(1).Format = '%.2f' ;
dtt.DataTipRows(2).Label = 'TPR' ;
dtt.DataTipRows(2).Format = '%.2f' ;
row = dataTipTextRow('Thresh',T(:,1),'%.2f') ;
dtt.DataTipRows(end+1) = row;
%% 
% 
% 
% Point confidence intervals were also calculated by |perfcurve| using bootstrapping 
% and these allow us to give the confidence intervals for the AUC.

% title(['AUC ',num2str(AUC(1),2) ,' (', num2str(AUC(2),2), ' ', num2str(AUC(3),2), ')'])
%% 
% 
% 
% In practice, we may choose an operating point aiming for a certain sensitivity. 
% Here we find the closest point on the curve to the 90% true positive rate (sensitivity) 
% and then plot the corresponding confidence intervals.

threshs = linspace(min(T(:,1)), max(T(:,1)),200) ;

ylbox = yline(axbox, threshs(end), 'LineWidth',2 ) ;
ylhist =yline(axhist,threshs(end) , 'LineWidth',2 ) ;
dt = datatip(hp,'DataIndex',length(threshs)) ;

% xlim(axroc,[0 1.5])
% ylim(axroc,[-0.2 1.02])
% yticks(axroc,[0:0.2:1])
% xticks(axroc,[0:0.1:1])

clear IM
nImages = length(threshs) ;
for ithresh = nImages:-1:1
    this_thresh = threshs(ithresh) ;
    [~, loc] = min(abs(T(:,1)-this_thresh)) ;

    if exist('hpp','var') && isvalid(hpp)
        delete(hpp)
    end
    hpp = plot(X(loc,1),Y(loc,1),'ro','MarkerSize',10,'MarkerFaceColor',[1 0 0]) ;

    ylbox.Value = this_thresh ;
    ylhist.Value = this_thresh ;

    if isvalid(dt)
        delete(dt) ;
    end
    dt = datatip(hp,'DataIndex',loc,'FontSize',12) ;

    drawnow
    F = getframe(hf) ;
    IM{nImages-ithresh+1} = frame2im(F) ;
    %pause(0.1)
end

fn = 'test.gif' ;

for idx = 1:nImages
    [A,map] = rgb2ind(IM{idx},256) ;
    if idx == 1
        imwrite(A,map,fn,'gif','LoopCount',Inf,'DelayTime',0.1);
    else
        imwrite(A,map,fn,'gif','WriteMode','append','DelayTime',0.1);
    end
end




% © 2021 David Atkinson
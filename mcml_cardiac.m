data_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
x = trials_results(4,:); % index 4 is the trial that worked best for cross-validation of MCML
x = x.';
x = [x, demo(:,[1, 2, 4])];
%%% zscore enforces consistent results when using the same configuration
%%% but produces higher volatility when changing nuclear hypertrophy scores
x = [normalize(x(:,1)),x(:,2),x(:,3:4)];

%%% Nuclear Hypertrophy labels
NHlabels = [1, 1, -1, 1, -1, 1, 1, 1, -1, -1];
%%% AF labels
AFlabels = [-1, 1, 1,-1, -1, 1, 1, 1, -1, 1];
trials = 1;
avg_acc = zeros(1,trials);
predics = zeros(1,10);
for jx = 1:trials
    fprintf("Trial %d \n", jx) 
    correct = 0;
	% uncomment the next line if not cross-validating MCML
    %mapping = mcml(x, AFlabels, 2);
    for ix = 1:size(data_set,2)
        training = data_set;
        testing = ix;
        training(testing) = [];
        trainlabels = AFlabels(training);
        xtrain = x(training,:);
        ytrain = AFlabels(training);
        if(AFlabels(ix)==-1 )
            %xtrain = [xtrain;xtrain(trainlabels==-1,:)];
            %ytrain = [ytrain,ytrain(trainlabels==-1)];
        end
		% The following chunk needs to be altered if not performing cross-validation on MCML
        mapping = mcml(xtrain, ytrain, 2);
        M = mapping.M;
        new_x = x*M;
		% If not performing cross-validation, simple change so that train_x = new_x, and likewise for train_y.
        train_x = new_x(training,:);
        train_y = AFlabels(training);
		% Balancing the training set so that there are same number of -1 and +1 examples
        if(AFlabels(ix)==-1)
            train_x = [train_x;train_x(trainlabels==-1,:)];
            train_y = [train_y,trainlabels(trainlabels==-1)];
        end
        test_x = new_x(testing,:);
        mod = fitcsvm(train_x,train_y,'KernelFunction','linear','BoxConstraint', 11);
        [pred,score] = predict(mod, test_x);
        predics(ix) = pred;
        if(pred == AFlabels(ix))
            correct = correct + 1;
        else
            disp(ix);
            figure;
            hold on
            colors = AFlabels;
            colors(testing) = 0;
            gscatter(new_x(:,1), new_x(:,2), colors, 'rbg', 'o**', [], 'off');
            W = [mod.Beta;mod.Bias];
            plotline(W,'k');
        end
    end
    acc = correct/size(data_set,2);
    avg_acc(jx) = acc;
end
avg_acc = sum(avg_acc)/trials;
fprintf("Accuracy = %d%% \n", 100*avg_acc) 
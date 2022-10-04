%%
close all
clear variables​
saveFig = false;

plotIndi = false;
% BVs = [16, 36, 64, 100, 144];
BVs = 100;
figureRoot = '/tmp';

for b = 1:length(BVs)
    BV = BVs(b);

    name = [num2str(BV) 'BV/'];
    root = ['/home/rbain/git/MSTd_AE_EA/evolve_MST_SNN_model_CARLsim/results/' name];
    result_dir = dir(root);  % get a list of all files and folders
    folders = {result_dir.name};  % put the names into cell array
    % folders = folders([result_dir.isdir] & ~startsWith(folders, '.')); 
    folders = folders([result_dir.isdir])
    combFitness = false;

%%
    formatSpec = 'popFitness-%d: %f\n';
    numFitness = 1;
    if combFitness
    %     formatSpec = 'fitness-%d: %f\npopFitness-%d: %f\nnetworkFitness-%d: %f\n';
        formatSpec = 'fitness-%d: %f\npopFitness-%d: %f\n';
        fitnessInd = 2;
        numFitness = 2;
    end
    sizeFitness = [2 Inf];
    indiNum = 50;
    numNetwork = length(folders);
    maxRuns = 20;
    fitnessMaxAll = zeros(numNetwork,maxRuns);

    % subplot(2,1,1)
    for n=1:numNetwork
        d = folders(n);
        % load all fitness, row#1:generation, row#2:fitness
        file = fullfile(root, d, 'fitness.txt');
        fileID = fopen(string(file),'r');
        fitness = fscanf(fileID, formatSpec, sizeFitness);

        if size(fitness,2) > indiNum*maxRuns
            fitness = fitness(:,size(fitness,2)-indiNum*maxRuns:end);
        end
        % group fitness into individuals
        try
            fitnessGrouped = reshape(fitness(2,:), numFitness, indiNum, []);
        catch exception
            newInd = floor(size(fitness,2)/indiNum) * indiNum;
            fitnessGrouped = reshape(fitness(2,1:newInd), numFitness, indiNum, []);
        end
        if combFitness
            fitnessGrouped = fitnessGrouped(fitnessInd,:,:);
        end

        fitnessGrouped = reshape(fitnessGrouped(1,:,:),indiNum,[]);
        numGenerations = size(fitnessGrouped,2);
        fitnessAll((n-1)*indiNum+1:n*indiNum,:) = fitnessGrouped;

        fitnessMax = max(fitnessGrouped, [], 1);
        fitnessMax(fitnessMax < -10) = -1; % correlation coef is between -1 and 1
        
        % find best so far fitness
        for i = 2:numGenerations
            if fitnessMax(i-1) > fitnessMax(i)
                fitnessMax(i) = fitnessMax(i-1);
            end
        end    
        fitnessMaxAll(n,:) = fitnessMax;
        if plotIndi
            errorbar(1:maxRuns, fitnessMax,std(fitnessGrouped,1))
            plot(fitnessMax, 'LineWidth', 1.5)
            xlabel('Generation')
            ylabel('Fitness')
            title(['max: ' num2str(max(fitnessMax))])
            axis([0 maxRuns 0 1])
            hold on
            pause()
        end
    end
    % set(gca,'FontSize',18)
    fitnessAllBV{b} = fitnessMaxAll;

end
set(gca,'FontSize',18)
%% plot mean and CI

ci = 0.95 ;
alpha = 1 - ci;

% subplot(2,1,2)
options.color_area = [243 169 114]./255;    % Orange theme
options.color_line = [236 112  22]./255;
options.alpha      = 0.5;
options.line_width = 2;
options.x_axis = 1:maxRuns;
options.x_axis = options.x_axis(:);

for b = 1:length(BVs)
    BV = BVs(b);
    name = [num2str(BV) 'BV'];
    data = fitnessAllBV{b};
    data_mean = mean(data,1);
    % data_max = max(data,1);
    % data_min = min(data,1);
    data_std  = std(data,0,1);
    error = (data_std./sqrt(size(data,1))).*1.96; %ci95

    h = figure(1);
    % Plotting the result
    x_vector = [options.x_axis', fliplr(options.x_axis')];
    patch = fill(x_vector(), [data_mean+error,fliplr(data_mean-error)], options.color_area);
    % patch = fill(x_vector, [data_max, fliplr(data_min)], options.color_area);
    set(patch, 'edgecolor', 'none');
    set(patch, 'FaceAlpha', options.alpha);

    hold on;
    plot(options.x_axis, data_mean, 'color', options.color_line, ...
        'LineWidth', options.line_width);
    xlim([0 maxRuns])
    ylim([0 1])
    xlabel('Generation')
    ylabel('Fitness')
    set(gca,'ytick',[0:0.1:1])
    set(gca,'FontSize',18)
    title(name)
    grid on
    h.Position = [1 1 580 380];
    hold off

    if saveFig
        saveas(h,[figureRoot, name '.svg'])
    end
end
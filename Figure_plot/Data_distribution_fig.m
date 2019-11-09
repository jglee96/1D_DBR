clear;
close all;

%%
Data_dist = csvread('Data_distribution(02).csv');
xr = Data_dist(1, :);
Treward = Data_dist(2, :);
xc = Data_dist(3, :);
Tcost = Data_dist(4, :);
colsWithZeros = any(xc==0, 1);
xc = xc(:, ~colsWithZeros);
Tcost = Tcost(:, ~colsWithZeros);

%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% reward
figure(1);
bar(xr, Treward./1000);
xlabel('Reward', 'Fontsize', label_Fontsize);
ylabel('# of data (x1000)', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

% cost
figure(2);
bar(xc, Tcost./1000);
xlabel('Cost', 'Fontsize', label_Fontsize);
ylabel('# of data (x1000)', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

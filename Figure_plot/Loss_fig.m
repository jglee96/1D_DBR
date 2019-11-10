clear;
close all;

Training_loss = csvread('ANN_Training_loss.csv');
Test_loss = csvread('ANN_Test_loss.csv');

figure(1);
plot(Training_loss, 'k', 'Linewidth', 2);
hold on;
plot(Test_loss, 'b', 'Linewidth', 2);
xlabel('Step', 'Fontsize', 20);
ylabel('Loss', 'Fontsize', 20);
ylim([0 0.1]);
xlim([0 100]);
ax = gca;
ax.FontSize = 15;

figure(2);
plot(Training_loss, 'k', 'Linewidth', 2);
hold on;
plot(Test_loss, 'b', 'Linewidth', 2);
xlabel('Step', 'Fontsize', 20);
ylabel('Loss', 'Fontsize', 20);
ax = gca;
ax.FontSize = 15;
clear;
close all;

Training_loss = csvread('ANN_Training_loss.csv');
Test_loss = csvread('ANN_Test_loss.csv');

lw = 3;
fs = 20;

figure(1);
plot(Training_loss, 'k', 'Linewidth', lw);
hold on;
plot(Test_loss, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ylim([0 0.1]);
xlim([0 100]);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';

figure(2);
plot(Training_loss, 'k', 'Linewidth', lw);
hold on;
plot(Test_loss, 'b', 'Linewidth', lw);
xlabel('Step', 'Fontsize', fs);
ylabel('Loss', 'Fontsize', fs);
ax = gca;
ax.FontSize = 15;
ax.FontWeight = 'bold';
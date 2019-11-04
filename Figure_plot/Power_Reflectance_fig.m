clear;
close all;

c = 299792458;
%% ARL %%
ARL_wave = csvread('ARL_wave.csv');
ARL_freq = csvread('ARL_freq.csv');
ARL_E_low = csvread('ARL_E_low.csv');
ARL_E_high = csvread('ARL_E_high.csv');
ARL_Rmax = max(ARL_freq(2,:));
% ARL find max reflectance information
ARL_Rmax_waveindex = find(ARL_wave(2,:) == ARL_Rmax);
ARL_Rmax_wave = ARL_wave(1, ARL_Rmax_waveindex);
ARL_Rmax_freqindex = find(ARL_freq(2,:) == ARL_Rmax);
ARL_Rmax_freq = ARL_freq(1, ARL_Rmax_freqindex);
% ARL polt properties
ARL_Linewidth = 2;
ARL_color = 'b';

%% ANN %%
ANN_wave = csvread('ANN_wave.csv');
ANN_freq = csvread('ANN_freq.csv');
ANN_E_low = csvread('ANN_E_low.csv');
ANN_E_high = csvread('ANN_E_high.csv');
ANN_Rmax = max(ANN_freq(2,:));
% ARL find max reflectance information
ANN_Rmax_waveindex = find(ANN_wave(2,:) == ANN_Rmax);
ANN_Rmax_wave = ANN_wave(1, ANN_Rmax_waveindex);
ANN_Rmax_freqindex = find(ANN_freq(2,:) == ANN_Rmax);
ANN_Rmax_freq = ANN_freq(1, ANN_Rmax_freqindex);
% ARL polt properties
ANN_Linewidth = 2;
ANN_color = 'r.-';

%% Theory %%
Theory_wave = csvread('Theory_wave.csv');
Theory_freq = csvread('Theory_freq.csv');
Theory_E_low = csvread('Theory_E_low.csv');
Theory_E_high = csvread('Theory_E_high.csv');
Theory_Rmax = max(Theory_freq(2,:));
% Theory find max reflectance information
Theory_Rmax_waveindex = find(Theory_wave(2,:) == Theory_Rmax);
Theory_Rmax_wave = Theory_wave(1, Theory_Rmax_waveindex);
Theory_Rmax_freqindex = find(Theory_freq(2,:) == Theory_Rmax);
Theory_Rmax_freq = Theory_freq(1, Theory_Rmax_freqindex);
% Theory polt properties
Theory_Linewidth = 2;
Theory_color = 'k';

%% Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% wavelength
figure(1);
plot(ARL_wave(1,:), ARL_wave(2,:), ARL_color, 'Linewidth', ARL_Linewidth);
hold on;
plot(ANN_wave(1,:), ANN_wave(2,:), ANN_color, 'Linewidth', ANN_Linewidth);
hold on;
plot(Theory_wave(1,:), Theory_wave(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

% frequency
figure(2);
%plot(ARL_freq(1,:), ARL_freq(2,:), ARL_color, 'Linewidth', ARL_Linewidth);
%hold on;
%plot(ANN_freq(1,:), ANN_freq(2,:), ANN_color, 'Linewidth', ANN_Linewidth);
%hold on;
plot(Theory_freq(1,:), Theory_freq(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Frequency [THz]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

%% Plot E-field %%
design_wave = 300;
dx = 5;
nh = 2.092;
nl = 1;
th = design_wave/(4*nh);
tl = design_wave/(4*nl);

ARL_list = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, ...
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ...
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
ANN_list = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ...
            1, 1, 1, 1, 1, 1, 0, 0, 0, 0, ...
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ...
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, ...
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1];
% ARL Efield
figure(3);
plot(ARL_E_low(1,:), ARL_E_low(2,:), 'r', 'Linewidth', ARL_Linewidth);
hold on;
plot(ARL_E_high(1,:), ARL_E_high(2,:), 'b', 'Linewidth', ARL_Linewidth);
for i=2:80
    if ARL_list(i) ~= ARL_list(i-1)
        hold on;
        xline((i-1)*dx);
    end
end
xlabel('x [um]', 'Fontsize', label_Fontsize);
ylabel('E-field', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

% ANN Efield
figure(4);
plot(ANN_E_low(1,:), ANN_E_low(2,:), 'r', 'Linewidth', ANN_Linewidth);
hold on;
plot(ANN_E_high(1,:), ANN_E_high(2,:), 'b', 'Linewidth', ANN_Linewidth);
for i=2:80
    if ANN_list(i) ~= ANN_list(i-1)
        hold on;
        xline((i-1)*dx);
    end
end
xlabel('x [um]', 'Fontsize', label_Fontsize);
ylabel('E-field', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;

%Theory Efield
figure(5);
plot(Theory_E_low(1,:), Theory_E_low(2,:), 'r', 'Linewidth', Theory_Linewidth);
hold on;
plot(Theory_E_high(1,:), Theory_E_high(2,:), 'b', 'Linewidth', Theory_Linewidth);
t = 0;
for i=1:9
    if mod(i,2) == 1
        t = t+th;
        xline(t);
    elseif mod(i,2) == 0
        t = t+tl;
        xline(t);
    end
end
            
xlabel('x [um]', 'Fontsize', label_Fontsize);
ylabel('E-field', 'Fontsize', label_Fontsize);
ax = gca;
ax.FontSize = tick_Fontsize;
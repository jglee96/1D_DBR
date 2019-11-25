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
% ARL find bandwidth
ARL_waveFWHN = find_bandwidth(ARL_wave(:,:), ARL_Rmax_waveindex, 0.5);
ARL_freqFWHM = find_bandwidth(ARL_freq(:,:), ARL_Rmax_freqindex, 0.5);
ARL_wave99 = find_bandwidth(ARL_wave(:,:), ARL_Rmax_waveindex, 0.99);
ARL_freq99 = find_bandwidth(ARL_freq(:,:), ARL_Rmax_freqindex, 0.99);
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
% ANN find bandwidth
ANN_waveFWHN = find_bandwidth(ANN_wave(:,:), ANN_Rmax_waveindex, 0.5);
ANN_freqFWHM = find_bandwidth(ANN_freq(:,:), ANN_Rmax_freqindex, 0.5);
ANN_wave99 = find_bandwidth(ANN_wave(:,:), ANN_Rmax_waveindex, 0.99);
ANN_freq99 = find_bandwidth(ANN_freq(:,:), ANN_Rmax_freqindex, 0.99);
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
% ARL find bandwidth
Theory_waveFWHN = find_bandwidth(Theory_wave(:,:), Theory_Rmax_waveindex, 0.5);
Theory_freqFWHM = find_bandwidth(Theory_freq(:,:), Theory_Rmax_freqindex, 0.5);
Theory_wave99 = find_bandwidth(Theory_wave(:,:), Theory_Rmax_waveindex, 0.99);
Theory_freq99 = find_bandwidth(Theory_freq(:,:), Theory_Rmax_freqindex, 0.99);
% Theory polt properties
Theory_Linewidth = 2;
Theory_color = 'k';

%% ARL Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% wavelength
figure;
plot(ARL_wave(1,:), ARL_wave(2,:), ARL_color, 'Linewidth', ARL_Linewidth);
hold on;
plot(Theory_wave(1,:), Theory_wave(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
title('ARL');
xlim([150 3000]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

% frequency
figure;
plot(ARL_freq(1,:), ARL_freq(2,:), ARL_color, 'Linewidth', ARL_Linewidth);
hold on;
plot(Theory_freq(1,:), Theory_freq(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Frequency [THz]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
% title('ARL');
xlim([0.1 2]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

figure;
plot(ARL_freq(1,:), ARL_freq(2,:), ARL_color, 'Linewidth', ARL_Linewidth);
hold on;
plot(Theory_freq(1,:), Theory_freq(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Frequency [THz]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
% title('ARL');
xlim([0.8 1.2]);
ylim([0.99 1]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';
%% ANN Plot %%
% plot properties
label_Fontsize = 20;
tick_Fontsize = 15;

% wavelength
figure;
plot(ANN_wave(1,:), ANN_wave(2,:), ANN_color, 'Linewidth', ANN_Linewidth);
hold on;
plot(Theory_wave(1,:), Theory_wave(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Wavelength [um]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
title('ANN');
xlim([150 3000]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

% frequency
figure;
plot(ANN_freq(1,:), ANN_freq(2,:), ANN_color, 'Linewidth', ANN_Linewidth);
hold on;
plot(Theory_freq(1,:), Theory_freq(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Frequency [THz]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
% title('ANN');
xlim([0.1 2]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';

figure;
plot(ANN_freq(1,:), ANN_freq(2,:), ANN_color, 'Linewidth', ANN_Linewidth);
hold on;
plot(Theory_freq(1,:), Theory_freq(2,:), Theory_color, 'Linewidth', Theory_Linewidth);
xlabel('Frequency [THz]', 'Fontsize', label_Fontsize);
ylabel('Reflectance', 'Fontsize', label_Fontsize);
% title('ANN');
xlim([0.8 1.3]);
ylim([0.99 1]);
ax = gca;
ax.FontSize = tick_Fontsize;
ax.FontWeight = 'bold';
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
figure;
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
title('ARL E-field');
ax = gca;
ax.FontSize = tick_Fontsize;

% ANN Efield
figure;
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
title('ANN E-field');
ax = gca;
ax.FontSize = tick_Fontsize;

%Theory Efield
figure;
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
title('Theory E-field');
ax = gca;
ax.FontSize = tick_Fontsize;

function y=find_bandwidth(R, max_idx, r)
for i=1:1:length(R)
    if R(2, max_idx+i) < R(2, max_idx)*r
        h_idx = max_idx+i;
        break
    end
end

for i=1:1:length(R)
    if R(2, max_idx-i) < R(2, max_idx)*r
        l_idx = max_idx-i;
        break
    end
end
y = abs(R(1, h_idx) - R(1, l_idx));
end
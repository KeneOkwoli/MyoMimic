% MyoMimic Real-Time EMG Plotter for MATLAB (5 Channels)

clear; clc; close all;

%% CONFIGURATION
COM_PORT = 'COM10';        
BAUD_RATE = 115200;
PLOT_WINDOW = 500;
UPDATE_RATE = 0.05;

%% SETUP SERIAL CONNECTION
fprintf('Connecting to ESP32 on %s...\n', COM_PORT);

try
    s = serialport(COM_PORT, BAUD_RATE);
    configureTerminator(s, "LF");
    fprintf('✓ Connected!\n');
catch
    error('Failed to connect. Check COM port and close Arduino Serial Monitor');
end

pause(2);
flush(s);

%% INITIALISE PLOT
fig = figure('Name', 'MyoMimic - Real-Time EMG (5 Channels)', ...
             'NumberTitle', 'off', ...
             'Position', [100 100 1200 800]);

channel_names = {'CH0-Dorsal', 'CH1-Radial', 'CH2-Dorsal-Radial', 'CH3-Ventral', 'CH4-Ulnar'};

colors = [
    0.5 0 0.5;
    0 0.4470 0.7410;
    0 0.8 0;
    0.9290 0.6940 0.1250;
    0.8500 0.3250 0.0980
];

raw_lines = cell(5,1);
filt_lines = cell(5,1);

% Top subplot
subplot(2,1,1);
hold on;
for i = 1:5
    raw_lines{i} = animatedline('Color', colors(i,:), 'LineWidth', 1.5, ...
                                'DisplayName', channel_names{i}, ...
                                'MaximumNumPoints', PLOT_WINDOW);
end
title('RAW EMG Signals (with 50Hz noise)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('ADC Value');
grid on;
ylim([1000 2200]);  % Better range for your data
xlim([0 PLOT_WINDOW]);
legend('show', 'Location', 'northeast');
hold off;

% Bottom subplot
subplot(2,1,2);
hold on;
for i = 1:5
    filt_lines{i} = animatedline('Color', colors(i,:), 'LineWidth', 1.5, ...
                                 'DisplayName', channel_names{i}, ...
                                 'MaximumNumPoints', PLOT_WINDOW);
end
title('FILTERED EMG Envelopes (Noise Removed)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Sample Number');
ylabel('Filtered Value');
grid on;
ylim([0 400]);  % Better range for filtered signals
xlim([0 PLOT_WINDOW]);
legend('show', 'Location', 'northeast');
hold off;

drawnow;

%% DATA BUFFERS
sample_count = 0;

%% MAIN LOOP
fprintf('\nPlotting started! Perform gestures to see EMG signals.\n');
fprintf('Press Ctrl+C to stop.\n\n');

tic;
last_update = toc;

try
    while ishandle(fig)
        if s.NumBytesAvailable > 0
            line = readline(s);
            
            if ~contains(line, ',')
                continue;
            end
            
            data = str2double(split(line, ','));
            
            if length(data) == 10 && ~any(isnan(data))
                sample_count = sample_count + 1;
                
                raw_vals = data(1:5);
                filt_vals = data(6:10);
                
                % Add points using the sample count modulo window size
                % This creates the scrolling effect
                x_pos = mod(sample_count - 1, PLOT_WINDOW) + 1;
                
                for i = 1:5
                    addpoints(raw_lines{i}, x_pos, raw_vals(i));
                    addpoints(filt_lines{i}, x_pos, filt_vals(i));
                end
                
                if toc - last_update > UPDATE_RATE
                    drawnow;
                    last_update = toc;
                end
            end
        else
            pause(0.001);
        end
    end
    
catch ME
    if strcmp(ME.identifier, 'MATLAB:interrupt')
        fprintf('\nStopped by user.\n');
    else
        fprintf('\nError: %s\n', ME.message);
    end
end

%% CLEANUP
fprintf('Closing serial port...\n');
clear s;
fprintf('Done!\n');
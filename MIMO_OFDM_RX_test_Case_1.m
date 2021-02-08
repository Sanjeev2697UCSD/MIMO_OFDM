close all ; clear all;

MIMO_OFDM_TX

%% Correlate for LTS
LTS_CORR_THRESH=.8;
DO_APPLY_CFO_CORRECTION=1;
DO_APPLY_SFO_CORRECTION=0;
DO_APPLY_PHASE_ERR_CORRECTION=1;

DISP_RX_CONSTELLATION = 1;


% For simplicity, we'll only use RFA for LTS correlation and peak
% discovery. A straightforward addition would be to repeat this process for
% RFB and combine the results for detection diversity.

% Complex cross correlation of Rx waveform with time-domain LTS
lts_corr = abs(conv(conj(fliplr(lts_t)), sign(rx_vec_dec_1A)));

% Skip early and late samples - avoids occasional false positives from pre-AGC samples
lts_corr = lts_corr(32:end-32);

% Find all correlation peaks
lts_peaks = find(lts_corr > LTS_CORR_THRESH*max(lts_corr));

% Select best candidate correlation peak as LTS-payload boundary
% In this MIMO example, we actually have 3 LTS symbols sent in a row.
% The first two are sent by RFA on the TX node and the last one was sent
% by RFB. We will actually look for the separation between the first and the
% last for synchronizing our starting index.

[LTS1, LTS2] = meshgrid(lts_peaks,lts_peaks);
[lts_last_peak_index,y] = find(LTS2-LTS1 == length(lts_t));

% Stop if no valid correlation peak was found
if(isempty(lts_last_peak_index))
    fprintf('No LTS Correlation Peaks Found!\n');
    return;
end

% Set the sample indices of the payload symbols and preamble
% The "+32" here corresponds to the 32-sample cyclic prefix on the preamble LTS
% The "+192" corresponds to the length of the extra training symbols for MIMO channel estimation
mimo_training_ind = lts_peaks(max(lts_last_peak_index)) + 32;
payload_ind = mimo_training_ind + 192;

% Subtract of 2 full LTS sequences and one cyclic prefixes
% The "-160" corresponds to the length of the preamble LTS (2.5 copies of 64-sample LTS)
lts_ind = mimo_training_ind-160;

if(DO_APPLY_CFO_CORRECTION)
    %Extract LTS (not yet CFO corrected)
    rx_lts = rx_vec_dec_1A(lts_ind : lts_ind+159); %Extract the first two LTS for CFO
    rx_lts1 = rx_lts(-64 + [97:160]);
    rx_lts2 = rx_lts([97:160]);

    %Calculate coarse CFO est
    rx_cfo_est_lts = mean(unwrap(angle(rx_lts2 .* conj(rx_lts1))));
    rx_cfo_est_lts = rx_cfo_est_lts/(2*pi*64);
    
    cfo_offset   = angle(rx_lts2 * rx_lts1')/(2*pi*length(lts_t));
    
else
    rx_cfo_est_lts = 0;
end

% Apply CFO correction to raw Rx waveforms
rx_cfo_corr_t = exp(-1i*2*pi*cfo_offset*[0:length(rx_vec_dec_1A)-1]);
rx_dec_cfo_corr_1A = rx_vec_dec_1A .* rx_cfo_corr_t;
rx_dec_cfo_corr_1B = rx_vec_dec_1B .* rx_cfo_corr_t;
rx_dec_cfo_corr_1C = rx_vec_dec_1C .* rx_cfo_corr_t;
rx_dec_cfo_corr_1D = rx_vec_dec_1D .* rx_cfo_corr_t;


% MIMO Channel Estimatation
lts_ind_TXA_start = mimo_training_ind + 32 ;
lts_ind_TXA_end = lts_ind_TXA_start + 64 - 1;

lts_ind_TXB_start = mimo_training_ind + 32 + 64 + 32 ;
lts_ind_TXB_end = lts_ind_TXB_start + 64 - 1;

rx_lts_1A = rx_dec_cfo_corr_1A( lts_ind_TXA_start:lts_ind_TXA_end );
rx_lts_1B = rx_dec_cfo_corr_1B( lts_ind_TXA_start:lts_ind_TXA_end );
rx_lts_1C = rx_dec_cfo_corr_1C( lts_ind_TXA_start:lts_ind_TXA_end );
rx_lts_1D = rx_dec_cfo_corr_1D( lts_ind_TXA_start:lts_ind_TXA_end );

rx_lts_1A_f = fft(rx_lts_1A);
rx_lts_1B_f = fft(rx_lts_1B);

rx_lts_1C_f = fft(rx_lts_1C);
rx_lts_1D_f = fft(rx_lts_1D);

%% Perform Channel estimation 

H1A = rx_lts_1A_f ./ lts_f;
H1B = rx_lts_1B_f ./ lts_f;
H1C = rx_lts_1C_f ./ lts_f;
H1D = rx_lts_1D_f ./ lts_f;

payload_mat_1A = reshape(rx_dec_cfo_corr_1A(payload_ind:payload_ind + (N_SC + CP_LEN)*N_OFDM_SYMS-1),[], N_OFDM_SYMS);
payload_mat_1B = reshape(rx_dec_cfo_corr_1B(payload_ind:payload_ind + (N_SC + CP_LEN)*N_OFDM_SYMS-1),[], N_OFDM_SYMS);
payload_mat_1C = reshape(rx_dec_cfo_corr_1C(payload_ind:payload_ind + (N_SC + CP_LEN)*N_OFDM_SYMS-1),[], N_OFDM_SYMS);
payload_mat_1D = reshape(rx_dec_cfo_corr_1D(payload_ind:payload_ind + (N_SC + CP_LEN)*N_OFDM_SYMS-1),[], N_OFDM_SYMS);

%% Rx payload processing, Perform combining for 1X4 and 2X2 separately  

% Extract the payload samples (integral number of OFDM symbols following preamble)

% Remove the cyclic prefix
payload_mat_noCP_1A = payload_mat_1A(CP_LEN+[1:N_SC], :);
payload_mat_noCP_1B = payload_mat_1B(CP_LEN+[1:N_SC], :);
payload_mat_noCP_1C = payload_mat_1C(CP_LEN+[1:N_SC], :);
payload_mat_noCP_1D = payload_mat_1D(CP_LEN+[1:N_SC], :);

% Take the FFT
syms_f_mat_1A = fft(payload_mat_noCP_1A, N_SC, 1);
syms_f_mat_1B = fft(payload_mat_noCP_1B, N_SC, 1);
syms_f_mat_1C = fft(payload_mat_noCP_1C, N_SC, 1);
syms_f_mat_1D = fft(payload_mat_noCP_1D, N_SC, 1);


% syms_eq_mat_data_A =  syms_f_mat_A(SC_IND_DATA,:) .* (H11(SC_IND_DATA) + H12(SC_IND_DATA))';
% syms_eq_mat_data_B =  syms_f_mat_B(SC_IND_DATA,:) .* (H21(SC_IND_DATA) + H22(SC_IND_DATA))';

% received_vector = (syms_eq_mat_data_A + syms_eq_mat_data_B) ./ (abs(H11(SC_IND_DATA) + H12(SC_IND_DATA)).^2 + abs(H21(SC_IND_DATA) + H22(SC_IND_DATA)).^2).';

 
%*This is optional -- SFO correction*

% Equalize pilots
% Because we only used Tx RFA to send pilots, we can do SISO equalization
% here. This is zero-forcing (just divide by chan estimates)
syms_eq_mat_pilots_1A = syms_f_mat_1A ./ repmat(H1A.', 1, N_OFDM_SYMS);
syms_eq_mat_pilots_1B = syms_f_mat_1B ./ repmat(H1B.', 1, N_OFDM_SYMS);
syms_eq_mat_pilots_1C = syms_f_mat_1C ./ repmat(H1C.', 1, N_OFDM_SYMS);
syms_eq_mat_pilots_1D = syms_f_mat_1D ./ repmat(H1D.', 1, N_OFDM_SYMS);

if DO_APPLY_SFO_CORRECTION
    % SFO manifests as a frequency-dependent phase whose slope increases
    % over time as the Tx and Rx sample streams drift apart from one
    % another. To correct for this effect, we calculate this phase slope at
    % each OFDM symbol using the pilot tones and use this slope to
    % interpolate a phase correction for each data-bearing subcarrier.

	% Extract the pilot tones and "equalize" them by their nominal Tx values
 
	% Calculate the phases of every Rx pilot tone
 

	% Calculate the SFO correction phases for each OFDM symbol

    % Apply the pilot phase correction per symbol

else
	% Define an empty SFO correction matrix (used by plotting code below)
    pilot_phase_sfo_corr = zeros(N_SC, N_OFDM_SYMS);
end

%*This is optional* 
% Extract the pilots and calculate per-symbol phase error
if DO_APPLY_PHASE_ERR_CORRECTION
    pilots_f_mat_1A = syms_eq_mat_pilots_1A(SC_IND_PILOTS, :);
    pilots_f_mat_1B = syms_eq_mat_pilots_1B(SC_IND_PILOTS, :);
    pilots_f_mat_1C = syms_eq_mat_pilots_1C(SC_IND_PILOTS, :);
    pilots_f_mat_1D = syms_eq_mat_pilots_1D(SC_IND_PILOTS, :);
    pilot_phase_err_1A = angle(mean(pilots_f_mat_1A.*pilots_A));
    pilot_phase_err_1B = angle(mean(pilots_f_mat_1B.*pilots_A));
    pilot_phase_err_1C = angle(mean(pilots_f_mat_1C.*pilots_A));
    pilot_phase_err_1D = angle(mean(pilots_f_mat_1D.*pilots_A));

else
	% Define an empty phase correction vector (used by plotting code below)
    pilot_phase_err_1A = zeros(1, N_OFDM_SYMS);
    pilot_phase_err_1B = zeros(1, N_OFDM_SYMS);
    pilot_phase_err_1C = zeros(1, N_OFDM_SYMS);
    pilot_phase_err_1D = zeros(1, N_OFDM_SYMS);
end
pilot_phase_corr_1A = repmat(exp(-1i*pilot_phase_err_1A), N_SC, 1);
pilot_phase_corr_1B = repmat(exp(-1i*pilot_phase_err_1B), N_SC, 1);
pilot_phase_corr_1C = repmat(exp(-1i*pilot_phase_err_1C), N_SC, 1);
pilot_phase_corr_1D = repmat(exp(-1i*pilot_phase_err_1D), N_SC, 1);

% Apply pilot phase correction to both received streams
syms_f_mat_pc_1A = syms_f_mat_1A .* pilot_phase_corr_1A;
syms_f_mat_pc_1B = syms_f_mat_1B .* pilot_phase_corr_1B;
syms_f_mat_pc_1C = syms_f_mat_1C .* pilot_phase_corr_1C;
syms_f_mat_pc_1D = syms_f_mat_1D .* pilot_phase_corr_1D;



syms_eq_mat_data_1A =  syms_f_mat_pc_1A(SC_IND_DATA,:) .* H1A(SC_IND_DATA)';
syms_eq_mat_data_1B =  syms_f_mat_pc_1B(SC_IND_DATA,:) .* H1B(SC_IND_DATA)';
syms_eq_mat_data_1C =  syms_f_mat_pc_1A(SC_IND_DATA,:) .* H1C(SC_IND_DATA)';
syms_eq_mat_data_1D =  syms_f_mat_pc_1B(SC_IND_DATA,:) .* H1D(SC_IND_DATA)';

received_vector = (syms_eq_mat_data_1A + syms_eq_mat_data_1B + syms_eq_mat_data_1C + syms_eq_mat_data_1D) ./ (abs(H1A(SC_IND_DATA)).^2 + abs(H1B(SC_IND_DATA)).^2 + abs(H1C(SC_IND_DATA)).^2 + abs(H1D(SC_IND_DATA)).^2).';

% syms_eq_mat_data_A = syms_eq_mat_data_A .* pilot_phase_corr(SC_IND_DATA,:);
% syms_eq_mat_data_B = syms_eq_mat_data_B .* pilot_phase_corr(SC_IND_DATA,:);


% Perform combining for MIMO 1X4 and 2X2 
% you need to apply the MIMO equalization to each subcarrier separately and then perform combining

% payload_syms_mat_A = syms_eq_mat_A(SC_IND_DATA, :);
% payload_syms_mat_B = syms_eq_mat_B(SC_IND_DATA, :);

%% perform demodulate or demapping post combined symbols 

rx_syms_case_1 = reshape(received_vector, 1, []);

if(DISP_RX_CONSTELLATION > 0)
    % plot the demodulated output rx_syms_case_1 and rx_syms_case_2
    figure(4);
    scatter(real(rx_syms_case_1), imag(rx_syms_case_1),'filled');
    % scatter(real(rx_syms_case_1), imag(rx_syms_case_1),'filled');
    title(' Signal Space of received bits');
    xlabel('I'); ylabel('Q');
end

% figure(5);
% scatter(real(rx_syms_case_2), imag(rx_syms_case_2),'filled');
% title(' Signal Space of received bits');
% xlabel('I'); ylabel('Q');


% FEC decoder for the rx_syms_case_1 and rx_syms_case_2

Demap_out_case_1 = demapper(rx_syms_case_1,MOD_ORDER,1);
% Demap_out_case_2 = demapper(rx_syms_case_2,MOD_ORDER,1);


trel = poly2trellis(7, [171 133]);              % Define trellis

% viterbi decoder
rx_data_final_1= vitdec(Demap_out_case_1,trel,7,'trunc','hard');
% rx_data_final_2 = vitdec(Demap_out_case_2,trel,7,'trunc','hard');

% rx_data is the final output corresponding to tx_data, which can be used
% to calculate BER

[number,ber] = biterr(tx_data_a,rx_data_final_1);
% [number,ber] = biterr(tx_data_b,rx_data_final_2);

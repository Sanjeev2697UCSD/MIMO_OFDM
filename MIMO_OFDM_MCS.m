Index = 0;

MOD_ORDER               =  2;          % Modulation order in power of 2 (1/2/4/6 = BSPK/QPSK/16-QAM/64-QAM)

% DISP_RX_CONSTELLATION = 1;

% 1x4 Diversity SIMO
for Index = 1:10000
    for noise_index = 1:1:11
        MIMO_OFDM_RX_test_Case_1
        BER_1(Index*noise_index) = ber;

        MIMO_OFDM_RX_test_Case_2
        BER_2(Index*noise_index) = ber;
    end
end

[p1,x1] = histogram(BER_1); plot(x1,p1/sum(p1)); %PDF of 1X4 MIMO
[f1,x1] = ecdf(BER_1); plot(x1,f1); %CDF of 1x4 MIMO

[p2,x2] = histogram(BER_2); plot(x2,p2/sum(p2)); %PDF of 2X2 MIMO
[f2,x2] = ecdf(BER_2); plot(x2,f2); %CDF of 2X2 MIMO

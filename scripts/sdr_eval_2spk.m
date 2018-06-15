% requires bss_eval_sources.m from bss_eval tools
function sdr_eval_2spk(gt_spk1, gt_spk2, ev_spk1, ev_spk2, sdr_file)
    gt_spk1 = fopen(gt_spk1);
    gt_spk2 = fopen(gt_spk2);
    ev_spk1 = fopen(ev_spk1);
    ev_spk2 = fopen(ev_spk2);
    sdr_out = fopen(sdr_file, 'w');
    
    gt_spk1_cell = textscan(gt_spk1, '%s %s');
    gt_spk2_cell = textscan(gt_spk2, '%s %s');
    ev_spk1_cell = textscan(ev_spk1, '%s %s');
    ev_spk2_cell = textscan(ev_spk2, '%s %s');
    
    num_utts = length(gt_spk1_cell{1});
    sdr_tot  = 0;
    fprintf('Evaluate %d utterances...\n', num_utts);
    
    for uid = 1: num_utts
        if mod(uid, 100) == 0
            fprintf('Processed %d utterance...\n', uid);
        end
        gt_spk1_utt = audioread(gt_spk1_cell{2}{uid});
        gt_spk2_utt = audioread(gt_spk2_cell{2}{uid});
        ev_spk1_utt = audioread(ev_spk1_cell{2}{uid});
        ev_spk2_utt = audioread(ev_spk2_cell{2}{uid});
        
        gt = [gt_spk1_utt, gt_spk2_utt];
        ev = [ev_spk1_utt, ev_spk2_utt];
        [sdr, ~, ~, ~] = bss_eval_sources(ev', gt');
        fprintf(sdr_out, '%s\t%f\n', gt_spk1_cell{1}{uid}, mean(sdr));
        sdr_tot = sdr_tot + mean(sdr);
    end
    
    fprintf('Average SDR: %f\n', sdr_tot / num_utts);
end

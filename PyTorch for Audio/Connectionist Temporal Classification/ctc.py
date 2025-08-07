"""
A Torch Implementation of the CTC Loss!

All credit for code goes to vadimkantorov and the code at
https://github.com/vadimkantorov/ctc

This implementation here is a slightly simplified variant that 
is annotated to understand whats going on!

"""
import torch

NEG_INF = torch.finfo(torch.float32).min

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="none"):
    
    ### example: targets [28, 1, 17, 21] ###
    ### lob_probs.shape -> (T x B x C) ###
    ### input_lengths.shape -> (B, )
    ### target_lengths.shape -> (B, )
    
    seq_len, batch_size, num_classes = log_probs.shape
    B = torch.arange(batch_size)
  
    ### _t_a_r_g_e_t_s_: [28, 1, 17, 21, 0] -> Append a blank token at the end (we could really use anything, doesnt matter) ###
    ### its just a placeholder so our indexing is happy, we will never use this token anywhere or include it in the loss
    _t_a_r_g_e_t_s_ = torch.cat([targets, torch.zeros(batch_size, 1, device=log_probs.device, dtype=torch.long)], dim=-1)
    
    ### _t_a_r_g_e_t_s_: [0, 28, 0, 1, 0, 17, 0, 21, 0, 0] -> Insert blank tokens in between targets
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], dim=-1).flatten(start_dim=-2)

    ### Get flag for if we have two consecutive labels (AA) or different labels (AB) ###
    ### Prepend with [False, False] to keep the same shape as _t_a_r_g_e_t_s_, but False
    ### as its just a placeholder for the indexing to match! 

    ### Copmaring: 
    ### _t_a_r_g_e_t_s_[:, :-2] -> [0, 28, 0, 1, 0, 17, 0, 21]
    ### _t_a_r_g_e_t_s_[:, 2:]  -> [0, 1, 0, 17, 0, 21, 0, 0]
    ###                            [T, F, T, F, T, F, T, F]

    ### Prepend the Falses as a placeholder:
    ### [F, F] + [T, F, T, F, T, F, T, F] -> [F, F, T, F, T, F, T, F, T, F]
    diff_labels = torch.cat([torch.tensor([[False, False]], device=log_probs.device).expand(batch_size, -1), \
                                _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], dim=-1)

    ### Grab the log probability for the correct target at every timestep T ###
    ### First, expand _t_a_r_g_e_t_s_, repeating for all timesteps outputed by model logits:

    ### _t_a_r_g_e_t_s_.expand(seq_len, -1, -1) (let the seq_len of the model logits be 6)

    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|
    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|
    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|
    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|
    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|
    ### |0, 28, 0, 1, 0, 17, 0, 21, 0, 0|

    ### Then we gather the probability of these indexes from logprobs at every timestep along our logits ###
    ### log_probs.gather(dim=-1, index=_t_a_r_g_e_t_s_.expand(seq_len, -1, -1))

    ### |p(0|T=0), p(28|T=0), p(0|T=0), p(1|T=0), p(0|T=0), p(17|T=0), p(0|T=0), p(21|T=0), p(0|T=0), p(0|T=0)|
    ### |p(0|T=1), p(28|T=1), p(0|T=1), p(1|T=1), p(0|T=1), p(17|T=1), p(0|T=1), p(21|T=1), p(0|T=1), p(0|T=1)|
    ### |p(0|T=2), p(28|T=2), p(0|T=2), p(1|T=2), p(0|T=2), p(17|T=2), p(0|T=2), p(21|T=2), p(0|T=2), p(0|T=2)|
    ### |p(0|T=3), p(28|T=3), p(0|T=3), p(1|T=3), p(0|T=3), p(17|T=3), p(0|T=3), p(21|T=3), p(0|T=3), p(0|T=3)|
    ### |p(0|T=4), p(28|T=4), p(0|T=4), p(1|T=4), p(0|T=4), p(17|T=4), p(0|T=4), p(21|T=4), p(0|T=4), p(0|T=4)|
    ### |p(0|T=5), p(28|T=5), p(0|T=5), p(1|T=5), p(0|T=5), p(17|T=5), p(0|T=5), p(21|T=5), p(0|T=5), p(0|T=5)|

    log_probs_ = log_probs.gather(dim=-1, index=_t_a_r_g_e_t_s_.expand(seq_len, -1, -1))
    
    ### Create our log_alpha matrix that we will use for the dynamic programming calculation ###
    ### This will store, for every timestep in T, and sample in batch, what is the likelihood to get to ###
    ### a specific timestep along our targets ###

    ### Alpha is in the log space, we initialize with probability 0 (which for log is log(0) = -inf)
    ### also we add two extra indexes before our target length, so we can do the transition to the first
    ### timestep without breaking indexing 
    log_alpha = torch.full((seq_len, batch_size, 2 + _t_a_r_g_e_t_s_.shape[-1]), NEG_INF).to(log_probs.device)

    ### Initialize our log_alpha with the initial blank token log prob at the first token after the 2 extra paddings we 
    ### added, and the token predicted after the blanks tokens log prob at the text timestep. This is our starting point 
    ### and we can use dynamic programming to iterate through all timesteps ###
    log_alpha[0, :, 2] = log_probs[0, :, blank]
    log_alpha[0, :, 2+1] = log_probs[0, B, _t_a_r_g_e_t_s_[:,1]]
    print(log_alpha.shape)
    for T_ in range(1, T):

        ### Compute all possible ways to reach some position s in log_alpha[t, :, s] ###
        ### - Stay at s [log_alpha[t-1, :, s]]
        ### - move from s-1 [log_alpha[t-1, :, s-1]]
        ### - move from s-2 [log_alpha[t-1, :, s-2]] (if allowed by diff labels)
        ### sum these probabilities since any of these transitions are valid!
        ### We compute this in parallel though for all s, instead of one at a time
        
        ### Probabilities for the Current Timestep ###
        log_probs_T_ = log_probs_[T_]

        ### Probability of staying from the previous timestep ###
        log_alpha_T_prev_stay = log_alpha[T_-1, :, 2:]

        ### Probability of transitioning form the previous timestep ###
        log_alpha_T_prev_next = log_alpha[T_-1, :, 1:-1]

        ### Mask identifying valid transitions (Cant transition from A -> A without a blank token in between) ###
        ### This mask is the probabilities if we have a repeat and can skip a step ###
        ### If the t two steps back is different from the current t, it is a valid transition, so this is a flag
        ### that allows those probabilities to come in (a two step transition)
        log_alpha_two_step_transition = torch.where(diff_labels, input=log_alpha[T_-1, : , :-2], other=NEG_INF)

        ### Sum it all together! All our data is log, so we need to exponentiate to get probs, sum the probs, and log again ###
        ### Lets use torch.logsumexp to do this!
        prob = torch.logsumexp(torch.stack([log_alpha_T_prev_next, log_alpha_T_prev_stay, log_alpha_two_step_transition]), dim = 0)
        
        ### Add our probs of this transition to the previous T_ probs (log turns multiplication to sum) 
        ### store this in our log_alpha for the next T_ computation
        log_alpha[T_, :, 2:] = log_probs_T_ + prob

    ### We now have completed our Dynamic Programming! Lets wrap this up by looking at the log probabilities of the last timestep
    ### which has accumulated together the probabiliites of all the paths upto them. Now our input lengths for each sample can be
    ### different, so we only grab till the T of each samples length indicated in input_lengths (anything after would just be padding)
    final_log_alpha = log_alpha[input_lengths-1, B]

    ### Now that we have grabbed only till the input length (that came from the model with length T),
    ### We also only need upto the target length (however many tokens were in our target)
    ### Although, we inserted blanks to our original targets, so our target length is now doubled in size
    ### Also we added 2 tokens at the start, so we offset for them here as well
    
    ### Now a key idea! If our labels are: [28, 1, 17, 21]
    ### And then we added in blank tokens in between to create: [0, 28, 0, 1, 0, 17, 0, 21, 0, 28] (_t_a_r_g_e_t_s)
    ### Then our final label can either be 21 or 0 (remeber, the 28 we concatenated at the end was just filler we never touch it)
    ending_on_label_idx = 2 + target_lengths * 2 - 1
    ending_on_blank_idx = 2 + target_lengths * 2
    indexes_to_grab = torch.stack([ending_on_label_idx, ending_on_blank_idx], dim=-1)
    label_or_blank_ending_log_alphas = final_log_alpha.gather(dim=-1, index=indexes_to_grab)

    ### Now that we have both possibilities of endings, we have to ensure that we logsumexp up both together 
    ### As both are valid, so we want the total probability! Also we want negative log likelihood as we want to 
    ### maximize the probability (or minimize the negative probability)
    loss = - torch.logsumexp(label_or_blank_ending_log_alphas, dim=-1)
    
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:
        raise ValueError("Select reductions from 'none', 'sum' and 'mean' !")

if __name__ == "__main__":
    T, B, C = 128, 256, 32
    t = 50
    blank = 0
    device = "cuda"
    seed = 1
    atol = 1e-3

    logits = torch.randn(T, B, C).requires_grad_().to(device)
    targets = torch.randint(blank+1, C, (B,t), dtype=torch.long).to(device)
    input_lengths = torch.full((B,), T, dtype=torch.long).to(device)
    target_lengths = torch.full((B,), t, dtype=torch.long).to(device)
    log_probs = logits.log_softmax(dim=-1).to(device)

    torch_ctc = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0, reduction = 'none')
    torch_ctc_grad, = torch.autograd.grad(torch_ctc.mean(), logits, retain_graph = True)

    my_ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    my_ctc_grad, = torch.autograd.grad(my_ctc.mean(), logits, retain_graph = True)

    print('CTC Losses Match:', torch.allclose(torch_ctc, my_ctc, atol = atol))
    print('Grad matches:', torch.allclose(torch_ctc_grad, my_ctc_grad, atol = atol))
    
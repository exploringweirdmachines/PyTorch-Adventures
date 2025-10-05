"""
All credit goes to incredible work by Umar Jamil, definitely watch the video! 
https://github.com/hkproj/triton-flash-attention/blob/main/triton/flash_attention.py

And of course this is also based off of the official implementation provided by Triton!
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py

This adapts the existing code with Cupy!

"""
import triton
import triton.language as tl
import cupy as cp


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):

    if STAGE == 1:
        ### CAUSAL: I want indexes (for my K,V) that are upto the ###
        ### index of my queries. This way my queries only attend to ###
        ### Keys and values that are before it. 

        ### This applies to all K,V before the diagonal. These are all blocks 
        ### of queries, as long as we are before the diagonal i know for sure 
        ### that every KV must be less that my query. Lets say we have the 
        ### following output from our blockes QKT

        ### [qk00 qk01 qk02 qk03]
        ### [qk10 qk11 qk12 qk13]
        ### [qk20 qk21 qk22 qk23]
        ### [qk30 qk31 qk32 qk33]

        ### And each qk00 is a block of values (lets say 3 x 3). I know for sure
        ### that every value in qk10, qk20, qk21, qk30, qk31, qk32 dont break any 
        ### causality. every value in those specific blocks that queries are ###
        ### looking at k/v that are <= in index 

        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
        
    elif STAGE == 2:
        ### On the diagonal, we have another condition to handle:
        ### Lets say we grab the top left corner (qk00) and each block is processing 
        ### 3 queries and 3 keys. In our output:

        ### [x00 x01 x02]
        ### [x10 x11 x12]
        ### [x20 x21 x22]

        ### x01 x02 x12 are invalid positions as that would mean the query vector
        ### is attending to a vector after it So we just need to remove these extra 
        ### ones. This is just more post processing!

        ### The block we want is the one at the end of our completely valid Q blocks and the 
        ### next one after. So we essentialyl have a low/high containing onyl the single block 
        ### where Q ends! Its just in that diagonal we have to mask out half that block. 
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q

        ### compiler optimization, to tell it that lo is always a multiple of BLOCK_SIZE_Q 
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        ### In non-causal attention we attend to everything 
        lo, hi = 0, SEQ_LEN
    
    ### KV pointers are currently pointing at the very start of the Key/Value for this ###
    ### Specific batch and head. In the case of STAGE=1 or ELSE, we just start at 0. We will ###
    ### piece by piece load BLOCK_SIZE_KV sizes of our Keys nad Values and do our ops there ###
    ### but in STAGE=2, we only want to do the ops on the diagonal values, so we need to advance ###
    ### our index to there ###

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    ### Loop over our Ks and Vs ###
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        ### Let the compiler know that start_n is a multiple of BLOCK_N ###
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        ### Compute our QK (it was already pretransposed) ###
        K_block = tl.load(K_block_ptr)

        ### We have a (Q_BLOCK_SIZE x E) and (E x KV_BLOCK_SIZE) matricies ###
        ### we can just use dot to do our dot product computation ###
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            # Post process the diagonal 
            # off_q is the indexes of the queries we are processing
            # offs_kv is the indexes of the keys/values we are processing
            # we can offset our offs_kv for this specific iteration in the loop
            # and do a broadcast check to see for what spots every q is greater than our 
            # k positions.
            # 
            # [0]   [0 1 2 3] -> [True False False False]
            # [1]                [True True  False False]
            # [2]                [True True  True  False]
            # [3]                [True True  True  True]
            # and then we can just fill the False with a large negative number!

            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)

            ### Update our current estimate for the max ###
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))

            ### Now we can subtract our max from our values ###
            ### as we iteratively compute a stable softmax
            QK_block -= m_ij[:, None]
        else:
            ### There is nothing to specifically mask here anymore. As our limits 
            ### for k and v indexes are already set. If we are full attention, we dont
            ### really care, if we are causal then this part only handles the blocks that 
            ### dont have exceptions like STAGE=2
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        ### We subtracted the max (and masked if needed) now we exponentiate ###
        P_block = tl.math.exp2(QK_block)

        ### Compute the sum of the rows for this block ###
        l_ij = tl.sum(P_block, 1)

        ### Correction factor from the previous block so we can do an online softmax ###
        alpha = tl.math.exp2(m_i - m_ij)

        ### Apply the correction factor ###
        l_i = l_i * alpha + l_ij

        ### Load the Values ###
        V_block = tl.load(V_block_ptr)

        ### Cast ###
        P_block = P_block.to(tl.float16)

        ### Use our formuala to iteratively update our outputs O_new = PV + O_old * alpha ###
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        ### Update Estiamte for Next Iter ###
        m_i = m_ij

        ### Advance to next block ###
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        
    return O_block, l_i, m_i

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [16]
        for BLOCK_SIZE_KV in [16]
        for num_stages in ([3])
        for num_warps in [2]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  
    K, 
    V,  
    softmax_scale,
    M,  
    O, 
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    Q = tl.cast(Q, tl.pointer_type(tl.float16))
    K = tl.cast(K, tl.pointer_type(tl.float16))
    V = tl.cast(V, tl.pointer_type(tl.float16))
    M = tl.cast(M, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float16))

    ### our index batch head is just a flattened vector of our batch_size * number of heads ###
    ### this means if we want what batch we are on, we can divide by num heads ###
    ### if we want which head we are on we can use modulo ###
    index_batch_head = tl.program_id(1) 
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ### Compute our offset of where a particular batch and head starts ###
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    )
    
    ### Who likes pointer arithmetic? Remember, my Q data is:
    ### Q.shape = (BATCH x HEADS x SEQ_LEN x EMBED_DIM)
    ### Each thread will process a specific BATCH and HEAD as well as a BLOCK of our SEQ_LEN
    ### So I need ot basically do a Q[batch_idx, head_idx, start_q_idx:end_q_idx, :]
    ### To do this with pointer arithmetic it would kind of look like:

    # row_offset = block_index_q * BLOCK_SIZE_Q                ### starting query vector idx in block
    # col_offset = 0                                           ### no column offset as we want the entire embedding vector

    # for i in range(BLOCK_SIZE_Q):                            ### for every query index I want
    #     for j in range(HEAD_DIM):                            ### for the head I am in
    #         ptr = (Q + qkv_offset                            ### We want to start art the right batch/head starting point and then move over by the row/col offset
    #                 + (row_offset + i) * stride_Q_seq
    #                 + (col_offset + j) * stride_Q_dim)
    #         val = tl.load(ptr)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,                      ### Offset to the batch/head we are processing
        shape=(SEQ_LEN, HEAD_DIM),                ### Because we already indexed batch/head the shape that is left is just (SEQ_LEN, HEAD_DIM)
        strides=(stride_Q_seq, stride_Q_dim),     ### What are the strides of the remaining dimensions
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),### Indexes of the Block of queries we are processing
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     ### What is the shape our our block of queries?
        order=(1,0)                               ### Memory coalescing. We make our HEAD DIM in contiguous memory addresses for fast access over the embeddings
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),                
        strides=(stride_V_seq, stride_V_dim),     
        offsets=(0,0),                            ### When loading values we dont skip anything, as we will for loop over this in a bit
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),              
        order=(1,0)                               
    )

    ### Switching our strides transposes our matrix. Take for example
    ### [A B C]
    ### [D E F]
    
    ### This has a strides[0] of 3 and strides[1] of 1. Now in memory, its actually
    ### stored as [A B C D E F]
    
    ### So what if we make our stride[0] = 1 and stride[1] = 3?

    ### Starting at A, to get to the next column, we have to move over 3. So A next 
    ### to A you have D. To get to the next row you move over 1, so from A next next
    ### row would be B. And that is exactly the transpose if you keep going!
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,                      
        shape=(HEAD_DIM, SEQ_LEN),                ### Set shape to transpose dimension
        strides=(stride_K_dim, stride_K_seq),     ### invert the stride     
        offsets=(0,0),                           
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),              
        order=(0,1)                               ### We want contiguous memory along the HEAD_DIM first
    )
    
    ### Our output is the same as as queries, we only want to write to the indexes of the query vector block we loaded 
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),               
        strides=(stride_O_seq, stride_O_dim),     
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     
        order=(1,0)                               
    )

    ### Lets grab offsets to tell us which indexes of Queries we are processing ###
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    ### We also need our offsets for the kv, of how many kv vectors are we processing with 
    ### Every one of our query blocks? 
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    ### Running max initialized with -inf ###
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    ### Running sum for our denomiator (sum e^x) ###
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0 # We initialize with 1, because we take a log later so for stability

    ### Accumulation of our final qk^T v for our specific block of queries/keys/values ###
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    ### Load our Query Block ###
    Q_block = tl.load(Q_block_ptr)

    ### Inner loop ###
    ### Stage 3 is for causal
    if STAGE == 1 or STAGE == 3:

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i, 
            Q_block, 
            K_block_ptr, 
            V_block_ptr, 
            block_index_q, 
            softmax_scale, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV, 
            4 - STAGE, # if STAGE=3 -> 1 which does before diag. if STAGE=1 -> 3 which does full attention
            offs_q, 
            offs_kv, 
            SEQ_LEN
        )

    if STAGE == 3:

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i, 
            Q_block, 
            K_block_ptr, 
            V_block_ptr, 
            block_index_q, 
            softmax_scale, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV, 
            2,              # In causal attention we have to post process the diagonal values specifically
            offs_q, 
            offs_kv, 
            SEQ_LEN
        )

    ### Store this as we need it for logsumexp in the backward pass ###
    m_i += tl.math.log2(l_i)

    ### We need to normalize our O by the sum of all our exponentials ###
    O_block = O_block / (l_i[:, None] + 1e-6)

    ### Store our temporary output M ###
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    
    O = tl.cast(O, tl.pointer_type(tl.float16))
    dO = tl.cast(dO, tl.pointer_type(tl.float16))
    D = tl.cast(D, tl.pointer_type(tl.float32))

    ### which group of seq_len vectors are we working on ###
    block_idx_q = tl.program_id(0)
    offs_q = block_idx_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    ### Which batch and which head in that batch are we computing ###
    ### And then grad the entire embedding dim ###
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM) 

    ### Load a single block of BLOCK_SIZE_Q rows of O ###
    ### Remember, O has shape (Batch x Num Heads x Seq Len x Embed Dim)

    # The reason you see offs_q[:, None] and offs_dim[None, :] is for broadcasting. We want a block
    # that contains every embedding value for every query vector. So this produces a 2d set of poitners
    # where the first dimension is which query do we want, and the second is which embedding value do we want from that embedding vector (all of them in our case)

    O_block = tl.load(
        O  
        + index_batch_head * HEAD_DIM * SEQ_LEN # Each batch has HEAD_DIM * SEQ_LEN items inside so advance to our correct index
        + offs_q[:, None] * HEAD_DIM            # Get to the first query vector we want to process in this block 
        + offs_dim[None, :]                     # Get the entire embedding length for each
    ).to(tl.float32) 
    
    ### Similarly load our dO BLock ###
    dO_block = tl.load(
        dO  
        + index_batch_head * HEAD_DIM * SEQ_LEN 
        + offs_q[:, None] * HEAD_DIM            
        + offs_dim[None, :]                     
    ).to(tl.float32) 

    
    ### Compute our D_block ###
    D_block = tl.sum(dO_block * O_block, axis=1) # Creates a single scalar for EACH item in our block, so we get (BLOCK_SIZE_Q, )

    ### Store it ###
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):  
    """
    Identical to the code _attn_bwd_dk_dv, except we are now
    looping through chunks of K and V instead for some given chunk 
    of Q. And we return only the grad for Q
    """

    LN2: tl.constexpr = 0.6931471824645996

    Q = tl.cast(Q, tl.pointer_type(tl.float16))
    K = tl.cast(K, tl.pointer_type(tl.float16))
    V = tl.cast(V, tl.pointer_type(tl.float16))
    dO = tl.cast(dO, tl.pointer_type(tl.float16))
    dQ = tl.cast(dQ, tl.pointer_type(tl.float16))
    dK = tl.cast(dK, tl.pointer_type(tl.float16))
    dV = tl.cast(dV, tl.pointer_type(tl.float16))
    M = tl.cast(M, tl.pointer_type(tl.float32))
    D = tl.cast(D, tl.pointer_type(tl.float32))

    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)

    index_block_kv = tl.program_id(0)

    start_q = index_block_kv * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    
    offs_kv = tl.arange(0, BLOCK_KV)

    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    Di = tl.load(D + offs_q)

    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp2(QK_block - M_block)

        if STAGE == 3:
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)

        dS_block = P_block * (dP_block - Di[:, None])
        dS_block *= LN2
        dS_block = dS_block.to(tl.float16)
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq

    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)

@triton.jit
def _attn_bwd_dk_dv(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch,
    stride_head,
    stride_seq,
    stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    
    LN2: tl.constexpr = 0.6931471824645996

    Q = tl.cast(Q, tl.pointer_type(tl.float16))
    K = tl.cast(K, tl.pointer_type(tl.float16))
    V = tl.cast(V, tl.pointer_type(tl.float16))
    dO = tl.cast(dO, tl.pointer_type(tl.float16))
    dQ = tl.cast(dQ, tl.pointer_type(tl.float16))
    dK = tl.cast(dK, tl.pointer_type(tl.float16))
    dV = tl.cast(dV, tl.pointer_type(tl.float16))
    M = tl.cast(M, tl.pointer_type(tl.float32))
    D = tl.cast(D, tl.pointer_type(tl.float32))

    ### Get to the correct Batch and the correct Head ###
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ### stride_batch is how many elements do I need to move to get to the next batch 
    ### stride_head is how many elements do I need to move to get to the next head
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)

    ### M and D dont have the head dimension, they are just (B x Head x Seq_len) ###
    ### So their offset to the correct seq_len is easier. We could have also used 
    ### the stride here for M or D but again our index_batch_head already accounts for 
    ### the first two dims, so we just need to multiply by seq_len to advance 
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)    
    
    ### Now we advance all our points to this correct batch/head dimension ###
    Q += offset_batch_head
    K += offset_batch_head
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head

    ### Also advance our pointers in M and D ###
    M += offset_batch_head_seq
    D += offset_batch_head_seq

    ### We will fix kv, so lets load it ###
    ### Offset for our embedding dim for each head ###
    offs_dim = tl.arange(0, HEAD_DIM)

    ### which kv block are we processing? ###
    index_block_kv = tl.program_id(0)

    ### get offset to the start of that block ###
    start_kv = index_block_kv * BLOCK_KV

    ### Get all indexes for that block ###
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    ### initialize dV and dK for this block to accumulate into ###
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    ### Now grab out K and V ###
    ### K and V have already been advanced to the correct Batch and Head. We just ###
    ### need to get a 2d grid of points that tell us which steps in our seqlen, and 
    ### which embed dims we want (all of them in our case)
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    )

    ### For each block of our k/v how many queries do we want to load? Lets ###
    ### make an offset for that! ###
    offs_q = tl.arange(0, BLOCK_Q)

    ### Now a trick, we will access Q as a transposed array. This is because in our setup ###
    ### we treat queries as a column vector, but right now they are row vectors.
    # 
    # ok so lets pretend that offs_q = [0,1,2,3] and offs_dim = [0,1,2,3,4,5,6,7]
    # this means we have 4 queries we want, and each query has an embed dim of 8. 
    # 
    # If I do  offs_q[None, :], that adds a new dimension, making a 1 x 4 vector like so:
    # offs_q[None, :] = [[0,1,2,3]]
    # 
    # similarly, when i do offs_dim[:, None] it adds a new dimension making an 8 x 1 
    #
    #   [[0]
    #    [1]
    #    [2]
    #    [3]
    #    [4]
    #    [5]
    #    [6]
    #    [7]]
    #
    # And adding them together will broadcast and give:

    # [[0 1 2 3]
    #  [1 2 3 4]
    #  [2 3 4 5]
    #  [3 4 5 6]
    #  [4 5 6 7]
    #  [5 6 7 8]
    #  [6 7 8 9]
    #  [7 8 9 10]]

    # Notice our final index shape is 8 x 4, so our indexing for embeddings is along
    # the rows, and our indexing for different query vectors is along the columns. This #
    # this will give us a column view of our slice of query vectors 
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    
    ### We will access dO like normal (no need to tranpose) ###
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    ### Iterate over the sequence of dims ###
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        
        ### Load qT ###
        qT_block = tl.load(qT_ptrs)

        ### Load the logsumexp values for the queries in the current block ###
        ### this means we have to advance our offs_q ### 
        ### recall that m = max(x) + log(sum(exp(x)))

        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        
        ### Now we can compute our QK^T for this specific block ###
        ### but we are doing everything transposed already so we need to do ###
        ### (QK^T)^T = KQ^T
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block)

        ### Now lets do softmax without actually doing softmax ! ###
        ### This is one of the most important parts of the implementation! 
        ### What we want is the softmaxed output in this block. But that would
        ### require us to store the entire N x N matrix in memory. So can we instead
        ### Compute it on the fly? In the forward pass we did online softmax, but we 
        ### can avoid that too

        ### remember we have stored m, our absolute max + log(denominator) 
        ### for every row of our softmax These were computed in the forward 
        ### pass so we can avoid doing it again. 

        ### Recall softmax: P_ij = exp(S_ij) / sum(exp(S_i))
        ### but we want stable softmax so instead we do
        ### softmax: P_ij = exp(S_ij - max(S_i)) / sum_j(exp(S_ij - max(S_i))
        ### and we already have m as our max so we can say:
        ### softmax: P_ij = exp(S_ij - m_i) / sum_j(exp(S_ij - m_i))

        ### So, what happens if we do this:

        ### exp(QK^T - m) = exp(QK^T - max - log(denominator))
        ### = exp(QK^T - max) / exp(log(denominator))
        ### Isnt that just our softmax? yes! So we can get our softmax back really
        ### easily with this trick!!!
        P_T_block = tl.math.exp2(QK_T_block - m[None, :])

        ### Process for autoregressive masking. If we had a mask, then we dont need ### 
        ### grads from future steps to contibute, we can 0 them out ###
        if STAGE == 3:

            ### mask a 2d mask to check if our q values are greater than or equal to k ##
            ### These positions are valid ###
            mask_block = (offs_q[None, :] >= offs_kv[:, None])

            ### Fill our probabilties with 0 as they dont contribute ###
            P_T_block = tl.where(mask_block, P_T_block, 0.0)

        ### Lets load the dO block that correspond to the slice of queries we are currently ###
        ### processing. 
        dO_block = tl.load(dO_ptrs)

        ### Now we start to accumulate grads. Each block of the output contribute to our 
        ### gradient for dV. dV_i = sum_j(P_ij * dO_i)
        ### But we are not processing all of our sequence length at once, only chunks of it
        ### and our dV is dependent on contributions from the entire length so we can 
        ### just accumulate as we go for the correct positions we are processing
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)

        ### To get our dK we need D ###
        ### remember the formula for dK = sum_i (Pij * (dO^Tv_j - Di)q_i)
        ### and again we cant do that sum all at once, we will accumulate it up per block 

        ### Remember, that our K has already been transposed. All our formulas that go into 
        ### Saving our dK must also be transposed! so we can just 
        ### flip them all to be [dO^Tv_j]^T = v_T dO 
        Di = tl.load(D + offs_q)
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])

        ### Now remember we had scaled our forward by 1/ln(2) to accounts for our log(2) ###
        ## Now we need to undo that scaling! ###
        dS_T_block *= LN2

        dS_T_block = dS_T_block.to(tl.float16)

        ### All we have left then is the q_i and our constant scale to add in! ###
        # dK = sum_i (Pij * (dO^Tv_j - Di)q_i)
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

        ### Increment all our pointers ###
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs+= BLOCK_Q * stride_seq
    
    ### Store it for the specific batch, head, chunk of sequence we processed ###
    dv_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dv_block_ptrs, dV_block)

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)

def fused_sdpa_forward(Q, K, V, 
                       causal, 
                       softmax_scale=None):
    
    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    ### This Implementation Assumes fp16 so downcast if needed ###
    if Q.dtype != cp.float16:
        Q = Q.astype(cp.float16)
    if K.dtype != cp.float16:
        K = K.astype(cp.float16)
    if V.dtype != cp.float16:
        V = V.astype(cp.float16)

    ### Make sure there is contiguous memory layout ####
    if not Q.flags.c_contiguous:
        Q = cp.ascontiguousarray(Q)
    if not K.flags.c_contiguous:
        K = cp.ascontiguousarray(K)
    if not V.flags.c_contiguous:
        V = cp.ascontiguousarray(V)

    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM**0.5

    ### Instead of normal log ops we will do log2 as its more numerically stable. ###
    ### but that means we are off by a constant factor. log_a(x) = log_b(x) / log_b(a)
    ### by the change of base formula. So log_2(x) = ln(x) / ln(2) so our constant 
    ### factor is just 1 / ln(2) ~ 1.442...
    softmax_scale *= 1.44269504

    O = cp.empty_like(Q)
    stage = 3 if causal else 1

    grid = lambda args: (
        triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
        1,
    )

    # M is the logsumexp for the backward pass, one for each query
    # Make sure to create it on the right device as we are not using empty_like
    with cp.cuda.Device(Q.device.id):
        M = cp.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=cp.float32
        )

    _attn_fwd[grid](
        Q=Q.data.ptr,
        K=K.data.ptr,
        V=V.data.ptr,
        softmax_scale=softmax_scale,
        M=M.data.ptr,
        O=O.data.ptr,
        stride_Q_batch=Q.strides[0] // Q.itemsize,
        stride_Q_head=Q.strides[1] // Q.itemsize,
        stride_Q_seq=Q.strides[2] // Q.itemsize,
        stride_Q_dim=Q.strides[3] // Q.itemsize,
        stride_K_batch=K.strides[0] // K.itemsize,
        stride_K_head=K.strides[1] // K.itemsize,
        stride_K_seq=K.strides[2] // K.itemsize,
        stride_K_dim=K.strides[3] // K.itemsize,
        stride_V_batch=V.strides[0] // V.itemsize,
        stride_V_head=V.strides[1] // V.itemsize,
        stride_V_seq=V.strides[2] // V.itemsize,
        stride_V_dim=V.strides[3] // V.itemsize,
        stride_O_batch=O.strides[0] // O.itemsize,
        stride_O_head=O.strides[1] // O.itemsize,
        stride_O_seq=O.strides[2] // O.itemsize,
        stride_O_dim=O.strides[3] // O.itemsize,
        BATCH_SIZE=Q.shape[0],
        NUM_HEADS=Q.shape[1],
        SEQ_LEN=Q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
    )

    return Q, K, V, O, M

def fused_sdpa_backward(dO, 
                        Q, K, V, 
                        O, M, 
                        causal,
                        softmax_scale=None, ):

    ### Ensure our grads are contiguous ###
    if not dO.flags.c_contiguous:
        dO = cp.ascontiguousarray(dO)
    if not dO.dtype == cp.float16:
        dO = dO.astype(cp.float16)
    
    assert Q.strides == K.strides == V.strides == O.strides == dO.strides
    dQ = cp.empty_like(Q)
    dK = cp.empty_like(K)
    dV = cp.empty_like(V)

    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    NUM_WARPS, NUM_STAGES = 4, 3
    BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 16, 128
    
    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM**0.5

    softmax_scale *= 1.44269504

    preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
    D = cp.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

    # Compute all the elements Di
    _attn_bwd_preprocess[preprocess_grid](
        O=O.data.ptr,
        dO=dO.data.ptr,
        D=D.data.ptr,
        SEQ_LEN=SEQ_LEN,
        BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
        HEAD_DIM=HEAD_DIM,
    )

    grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)

    stage = 3 if causal else 1

    # Fix KV and iterate through all the Q blocks
    _attn_bwd_dk_dv[grid](
        Q=Q.data.ptr,
        K=K.data.ptr,
        V=V.data.ptr,
        softmax_scale=softmax_scale,
        dO=dO.data.ptr,
        dQ=dQ.data.ptr,
        dK=dK.data.ptr,
        dV=dV.data.ptr,
        M=M.data.ptr,
        D=D.data.ptr,
        stride_batch=Q.strides[0] // Q.itemsize,
        stride_head=Q.strides[1] // Q.itemsize,
        stride_seq=Q.strides[2] // Q.itemsize,
        stride_dim=Q.strides[3] // Q.itemsize,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        BLOCK_Q=BLOCK_SIZE_MICRO,
        BLOCK_KV=BLOCK_SIZE_MACRO,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Fix Q and iterate through all the KV block
    _attn_bwd_dq[grid](
        Q=Q.data.ptr,
        K=K.data.ptr,
        V=V.data.ptr,
        softmax_scale=softmax_scale,
        dO=dO.data.ptr,
        dQ=dQ.data.ptr,
        dK=dK.data.ptr,
        dV=dV.data.ptr,
        M=M.data.ptr,
        D=D.data.ptr,
        stride_batch=Q.strides[0] // Q.itemsize,
        stride_head=Q.strides[1] // Q.itemsize,
        stride_seq=Q.strides[2] // Q.itemsize,
        stride_dim=Q.strides[3] // Q.itemsize,
        NUM_HEADS=NUM_HEADS,
        SEQ_LEN=SEQ_LEN,
        BLOCK_Q=BLOCK_SIZE_MACRO,
        BLOCK_KV=BLOCK_SIZE_MICRO,
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return dQ, dK, dV

def test_cupy_op(BATCH_SIZE, 
                 NUM_HEADS, 
                 SEQ_LEN, 
                 HEAD_DIM, 
                 causal, 
                 dtype=cp.float16):
    
    Q = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    K = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    V = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    dO = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)

    Q_torch = torch.tensor(Q, requires_grad=True)
    K_torch = torch.tensor(K, requires_grad=True)
    V_torch = torch.tensor(V, requires_grad=True)
    dO_torch = torch.tensor(dO)

    softmax_scale = 1 / (HEAD_DIM**0.5)

    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
    P_torch = torch.matmul(Q_torch, K_torch.transpose(2, 3)) * softmax_scale
    if causal:
        P_torch[:, :, MASK == 0] = float("-inf")
    P_torch = torch.softmax(P_torch.float(), dim=-1).half()
    ref_O_torch = torch.matmul(P_torch, V_torch)
    ref_O_torch.backward(dO_torch)
    dQ_torch = Q_torch.grad
    dK_torch = K_torch.grad
    dV_torch = V_torch.grad

    ref_O_cupy = cp.array(ref_O_torch.detach().cpu().numpy())
    # ref_dq_cupy = cp.array(dQ_torch.detach().cpu().numpy())
    # ref_dk_cupy = cp.array(dK_torch.detach().cpu().numpy())
    # ref_dv_cupy = cp.array(dV_torch.detach().cpu().numpy())

    # triton implementation
    *_, tri_out, M = fused_sdpa_forward(Q, K, V, 
                                          causal=causal, 
                                          softmax_scale=softmax_scale)
    # dQ, dK, dV = fused_sdpa_backward(dO, Q, K, V, tri_out, M, 
    #                                   softmax_scale=softmax_scale, 
    #                                   causal=causal)

    # compare
    tol = 1e-2
    print("Max Out Diff:", cp.abs(ref_O_cupy - tri_out).max())
    # print("Max DQ Diff:", cp.abs(ref_dq_cupy - dQ).max())
    # print("Max DK Diff:", cp.abs(ref_dk_cupy - dK).max())
    # print("Max DV Diff:", cp.abs(ref_dv_cupy - dV).max())
    
    # assert cp.abs(ref_O_cupy - tri_out).max() < tol
    # assert cp.abs(ref_dq_cupy - dQ).max() < tol
    # assert cp.abs(ref_dk_cupy - dK).max() < tol
    # assert cp.abs(ref_dv_cupy - dV).max() < tol

def benchmark_flash_attention(
    BATCH_SIZE=1,
    NUM_HEADS=4,
    SEQ_LEN=128,
    HEAD_DIM=64,
    causal=False,
    dtype=cp.float16,
    warmup_iters=5,
    bench_iters=50,
):
    # Random inputs
    Q = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    K = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    V = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
    dO = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)

    Q_torch = torch.tensor(Q.get(), requires_grad=True, device="cuda", dtype=torch.float16)
    K_torch = torch.tensor(K.get(), requires_grad=True, device="cuda", dtype=torch.float16)
    V_torch = torch.tensor(V.get(), requires_grad=True, device="cuda", dtype=torch.float16)
    dO_torch = torch.tensor(dO.get(), device="cuda", dtype=torch.float16)

    softmax_scale = 1.0 / (HEAD_DIM**0.5)
    mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda")) if causal else None

    if causal:
        mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"))
        # Cast mask to same dtype as Q
        mask = mask.to(torch.float16)
    else:
        mask = None
        
    def torch_ref():
        P = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * softmax_scale
        if causal:
            P = P.masked_fill(mask == 0, float("-inf"))
        P = torch.softmax(P, dim=-1)
        O = torch.matmul(P, V_torch)
        return O

    def torch_sdpa():
        return torch.nn.functional.scaled_dot_product_attention(
            Q_torch, K_torch, V_torch, attn_mask=None if not causal else mask, is_causal=causal
        )

    def triton_sdpa():
        *_, tri_out, M, seed = fused_sdpa_forward(Q, K, V, causal, softmax_scale)
        return tri_out

    print("Running warmups...")
    for _ in range(warmup_iters):
        torch_ref()
        torch_sdpa()
        triton_sdpa()
    torch.cuda.synchronize()
    cp.cuda.Device().synchronize()

    def bench_fn(fn, sync_fn):
        start = time.time()
        for _ in range(bench_iters):
            out = fn()
        sync_fn()
        end = time.time()
        return (end - start) / bench_iters, out

    print("\nBenchmarking...")

    t_ref, ref_out = bench_fn(torch_ref, torch.cuda.synchronize)
    t_sdpa, sdpa_out = bench_fn(torch_sdpa, torch.cuda.synchronize)
    t_tri, tri_out = bench_fn(triton_sdpa, cp.cuda.Device().synchronize)

    print(f"PyTorch ref  : {t_ref*1000:.3f} ms")
    print(f"PyTorch SDPA : {t_sdpa*1000:.3f} ms")
    print(f"Triton CuPy  : {t_tri*1000:.3f} ms")

    ref_out_cu = cp.array(ref_out.detach().cpu().numpy())
    diff = cp.abs(ref_out_cu - tri_out)
    print(f"\nMax Out Diff: {diff.max():.4e}")

    return {
        "seq_len": SEQ_LEN,
        "torch_ref_ms": t_ref * 1000,
        "torch_sdpa_ms": t_sdpa * 1000,
        "triton_ms": t_tri * 1000,
        "max_diff": float(diff.max()),
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import time 

    print("Checking for Correctness")
    test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=15, HEAD_DIM=64, causal=True)
    # test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=128, HEAD_DIM=64, causal=False)
    print("All Correct!!")

    # print("Benchmarking")
    # all_results = []
    # for seq_len in [64,128,256,512,1024,2048,4096]:
    #     print("Testing Seq Len:", seq_len)
    #     results = benchmark_flash_attention(
    #         BATCH_SIZE=32,
    #         NUM_HEADS=12,
    #         SEQ_LEN=seq_len,
    #         HEAD_DIM=64,
    #         causal=True,
    #         dtype=cp.float16,
    #         bench_iters=50
    #     )
    #     print("SEQ_LEN:", seq_len)
    #     all_results.append(results)
    
    # seq_lens = np.array([r["seq_len"] for r in all_results])
    # forward_times = np.array([[r["torch_ref_ms"], r["torch_sdpa_ms"], r["triton_ms"]] for r in all_results])

    # plt.plot(seq_lens, forward_times[:, 0], "-o", label="PyTorch Naive", linewidth=3)
    # plt.plot(seq_lens, forward_times[:, 1], "-o", label="PyTorch SDPA", linewidth=3)
    # plt.plot(seq_lens, forward_times[:, 2], "-o", label="Triton FlashAttention", linewidth=3)
    # plt.xlabel("Sequence Length (L)")
    # plt.ylabel("Time (ms)")
    # plt.title("Forward Attention Runtime")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("benchmark/flash_attn.png")
    # plt.show()
    
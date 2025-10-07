"""
FlashAttention2 Kernel (Online-Softmax Applied to Attention)

Some really awesome resources that this was largely based off of!
1) Umar Jamil: https://github.com/hkproj/triton-flash-attention
2) Evintunador: https://github.com/evintunador/triton_docs_tutorials

And of course this is also based off of the official implementation provided by Triton!
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py

This adapts the existing code with Cupy!

1) NO Support for attention masks. This does CAUSAL or Not Causal. Custom masks are not supported (YET?)
2) NO Dropout on our attention scores! Maybe another feature to come?
3) FP16 support only. But we use flash attention to limit memory use, why would you train in fp32 if memory is a concern? 
   This also means that when training fp32 models, tensors will have to be downcasted!

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
    softmax_scale: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN,
):      
    """
    The inner loop of the forward flash attention method grabs a chunk of queries
    and loops through all the Keys/Values also in chunks, using online-softmax as we go
    """

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

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # Keys are transposed so SEQ dim is second
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    ### Loop over our Ks and Vs ###
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        ### Let the compiler know that start_n is a multiple of BLOCK_N ###
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        kv_indices = start_kv + offs_kv
        kv_padding_mask = kv_indices < SEQ_LEN

        ### Compute our QK (it was already pretransposed) ###
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")

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
            causal_mask = offs_q[:, None] >= kv_indices[None, :]
            mask = causal_mask & kv_padding_mask[None, :]
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, float("-inf"))

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
            QK_block = tl.where(kv_padding_mask[None, :], QK_block, float("-inf"))
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
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        ### Cast ###
        P_block = P_block.to(tl.float16)

        ### Use our formuala to iteratively update our outputs O_new = PV + O_old * alpha ###
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        ### Update Estimate for Next Iter ###
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
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in ([2,3,4])
        for num_warps in [4,8,16]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,  
    K, 
    V,  
    softmax_scale: tl.constexpr,
    M,  
    O, 
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_seq,
    stride_K_dim,
    stride_V_seq,
    stride_V_dim,
    stride_O_seq,
    stride_O_dim,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):  
    """
    Main forward method for Flash Attention, where for a block of queries
    we iteratively compute attention by looping over blocks of Keys/Values
    """

    ### When we do Q @ K, we use the tl.dot method to do this ###
    ### So the inner product loads a row/column of K and Q into ###
    ### registers for the actual computation where each row has HEAD_DIM elements ###
    ### So although we are chunking our sequence into BLOCK_SIZE_KV, we still need ###
    ### to load the entire embeddings. We want to make sure this isnt too large for ###
    ### efficiency. So we place a restriction here that our BLOCK_SIZE cannot be any ###
    ### larger than our HEAD_DIM. Id rather have more blocks scheduled to do less work ###
    ### than have fewer blocks each processing massive matricies for better GPU utilization ###
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    ### Cast our Pointers to the right type ###
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

    ### Intermediate data we store will be in a higher precision for efficiency ###
    ### Running max initialized with -inf ###
    m_i = tl.full(shape=[BLOCK_SIZE_Q], value=-1e6, dtype=tl.float32)

    ### Running sum for our denomiator (sum e^x) ###
    l_i = tl.full(shape=[BLOCK_SIZE_Q], value=1.0, dtype=tl.float32) # We initialize with 1, because we take a log later so for stability

    ### Accumulation of our final qk^T v for our specific block of queries/keys/values ###
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    ### Load our Query Block ###
    ### Now a super cool ability for block pointers. It can automagically check ###
    ### for invalid indexes (like if our Query we are indexing is greater than SEQ_LEN) ###
    ### And it will fill it with the padding option we give it! ###
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

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
        
        ### If we are doing causal attention, the blocks on the diagonal contain values that contain a transition. ###
        ### for example lets say we look at the top left block, and lets say each block is 4x4 ###

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### The issue is we are processing entire blocks at a time, but the elements of this block are not all valid. ###
        ### If we are causal, qk_01 for example, means query at time 0 is attending to a key in a future time 1. ###
        ### This breaks causality. So in the previous part, we already computed all the blocks upto the diagonal (if causal) ###

        ### Lets look at it at the block level, remember each block has 4x4 elements inside them:

        ### [B_00, B_01, B_02, B_03]
        ### [B_10, B_11, B_12, B_13]
        ### [B_20, B_21, B_22, B_23]
        ### [B_30, B_31, B_32, B_33]

        ### An in this case our B_00 block is the top left block from above.

        ### In the first stage (if causal) we can directly compute everything in 
        ### B_10, B_20, B_21, B_30, B_31, B_32

        ### Because its guaranteed that every query in that block is attending to keys that are before it
        ### But along our diagonal B_00, B_11, B_22, B_33, we have a transition inside the block
        
        ### Again for B_00 we had 

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### We need to compute the diagonal and this is easiest if we just do it separately and then 
        ### make sure to mask out the top triangle portion so we get 

        ### [qk_00, -inf , -inf , -inf ]
        ### [qk_10, qk_11, -inf , -inf ]
        ### [qk_20, qk_21, qk_22, -inf ]
        ### [qk_30, qk_31, qk_32, qk_33]

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
            2,              # In causal attention we have to post process the diagonal values specifically (STAGE=2)
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
    q_padding_mask = offs_q < SEQ_LEN
    tl.store(m_ptrs, m_i, mask=q_padding_mask)

    ### When storing our Output make sure to check boundary ###
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for num_stages in ([2,3,4])
        for num_warps in [4,8,16]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D,
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    
    """
    For the backward pass we need D = sum(o*dO). So lets just precompute
    it and store it!
    """

    ### Cast our Pointers ###
    O = tl.cast(O, tl.pointer_type(tl.float16))
    dO = tl.cast(dO, tl.pointer_type(tl.float16))
    D = tl.cast(D, tl.pointer_type(tl.float32))

    ### which group of seq_len vectors are we working on ###
    block_idx_q = tl.program_id(0)
    offs_q = block_idx_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    ### Create a mask for the sequence dimension ###
    q_padding_mask = offs_q >= 0
 
    ### Which batch and which head in that batch are we computing ###
    index_batch_head = tl.program_id(1)
    offs_dim = tl.arange(0, HEAD_DIM) 

    ### Index to specific Batch/Head/chunk of embeds we want ###
    batch_head_offset = index_batch_head * SEQ_LEN * HEAD_DIM
    O_ptrs = O + batch_head_offset + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]
    dO_ptrs = dO + batch_head_offset + offs_q[:, None] * HEAD_DIM + offs_dim[None, :]
    
    ### Load O block with padding mask ###
    O_block = tl.load(
        O_ptrs, 
        mask=q_padding_mask[:, None], 
        other=0.0
    ).to(tl.float32) 
    
    ### Load dO block with padding mask ###
    dO_block = tl.load(
        dO_ptrs,
        mask=q_padding_mask[:, None], 
        other=0.0
    ).to(tl.float32) 

    ### Compute our D_block ###
    D_block = tl.sum(dO_block * O_block, axis=1)

    ### Store D with padding mask ###
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block, mask=q_padding_mask)

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in ([2,3,4])
        for num_warps in [4,8,16]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
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
    start_kv = index_block_kv * BLOCK_SIZE_KV

    ### Get all indexes for that block ###
    offs_kv = start_kv + tl.arange(0, BLOCK_SIZE_KV)

    ### Create a mask so we dont access invalid KV Indices ###
    kv_mask = offs_kv < SEQ_LEN

    ### initialize dV and dK for this block to accumulate into ###
    dV_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    ### Now grab out K and V ###
    ### K and V have already been advanced to the correct Batch and Head. We just ###
    ### need to get a 2d grid of points that tell us which steps in our seqlen, and 
    ### which embed dims we want (all of them in our case)
    K_block = tl.load(
        K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim,
        mask=kv_mask[:, None],
        other=0.
    )
    V_block = tl.load(
        V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim,
        mask=kv_mask[:, None],
        other=0.
    )

    ### For each block of our k/v how many queries do we want to load? Lets ###
    ### make an offset for that! ###
    offs_q = tl.arange(0, BLOCK_SIZE_Q)

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
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_SIZE_Q) #SEQ_LEN // BLOCK_SIZE_Q
    if STAGE == 3:

        ### KV Can attend to any Q after it (as Q can only attend to stuff before it)
        ### so if we are autoregressive, then dont bother attending to Q values
        ### that are before these KVs, as that would mean the Q values are looking
        ### into the future

        ### Computes the earliest possible query position where the Q block could 
        ### potentially contribute to the current KV block.
        min_start_q = tl.maximum(0, start_kv - BLOCK_SIZE_Q + 1)

        ### Get the index of the first block that contains that starting position
        starting_block = tl.cdiv(min_start_q, BLOCK_SIZE_Q)

        ### Get the starting query of the block. It may be earlier than 
        ### our actual starting query, but we will mask any extra things later 
        ### But we need to stay aligned to block boundaries
        start_curr_q = starting_block * BLOCK_SIZE_Q

        ### Update the number of steps we need to take 
        num_steps = tl.cdiv(SEQ_LEN - start_curr_q, BLOCK_SIZE_Q)

        ### Advance our pointers forward to the correct starting point
        qT_ptrs += start_curr_q * stride_seq
        dO_ptrs += start_curr_q * stride_seq
        curr_q = start_curr_q

    for _ in range(num_steps):
        
        ### Update offs_q for current block ###
        offs_q = curr_q + tl.arange(0, BLOCK_SIZE_Q)

        ### Create a Mask for Q Blocks ###
        q_mask = offs_q < SEQ_LEN

        ### Load qT ###
        qT_block = tl.load(qT_ptrs, mask=q_mask[None, :], other=0.)

        ### Load the logsumexp values for the queries in the current block ###
        ### this means we have to advance our offs_q ### 
        ### recall that m = max(x) + log(sum(exp(x)))
        m = tl.load(M + offs_q, mask=q_mask, other=0.)
        
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

        ### Create a seq_len mask that masks out any invalid attention scores ###
        seq_mask = kv_mask[:, None] & q_mask[None, :]
        P_T_block = tl.where(seq_mask, P_T_block, 0.0)

        ### Lets load the dO block that correspond to the slice of queries we are currently ###
        ### processing. 
        dO_block = tl.load(dO_ptrs, mask=q_mask[:, None], other=0.)

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
        Di = tl.load(D + offs_q, mask=q_mask, other=0.)
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
        curr_q += BLOCK_SIZE_Q
        qT_ptrs += BLOCK_SIZE_Q * stride_seq
        dO_ptrs+= BLOCK_SIZE_Q * stride_seq
    
    ### Store it for the specific batch, head, chunk of sequence we processed ###
    dv_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dv_block_ptrs, dV_block, mask=kv_mask[:, None])

    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block, mask=kv_mask[:, None])

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64, 128]
        for num_stages in ([2,3,4])
        for num_warps in [4,8,16]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):  
    """
    Nearly identical to our _attn_bwd_dk_dv except now we 
    loop through our keys/values for a given set of queries
    """
    LN2: tl.constexpr = 0.6931471824645996

    ### Cast all of our pointers. M/D are our own temporary matricies we made ###
    ### to store data in, and we made it float32 so we keep it the same here! ###
    Q = tl.cast(Q, tl.pointer_type(tl.float16))
    K = tl.cast(K, tl.pointer_type(tl.float16))
    V = tl.cast(V, tl.pointer_type(tl.float16))
    dO = tl.cast(dO, tl.pointer_type(tl.float16))
    dQ = tl.cast(dQ, tl.pointer_type(tl.float16))
    dK = tl.cast(dK, tl.pointer_type(tl.float16))
    dV = tl.cast(dV, tl.pointer_type(tl.float16))
    M = tl.cast(M, tl.pointer_type(tl.float32))
    D = tl.cast(D, tl.pointer_type(tl.float32))

    ### Get what Batch/Head we are on ###
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(
        tl.int64
    )

    ### Advance everythign to that position ###
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

    ### Offsets to grab data along the embed dims ###
    offs_dim = tl.arange(0, HEAD_DIM)
    
    ### Which KV Block are we processing? This is held constant ####
    ### as we loop through our queries ###
    index_block_kv = tl.program_id(0)

    ### Get the starting position of the query block we want to process ###
    start_q = index_block_kv * BLOCK_SIZE_Q
    offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)

    ### Make sure to mask out any invalid queries ###
    q_mask = offs_q < SEQ_LEN

    ### Grab our block of queries ###
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim, 
                      mask=q_mask[:, None], 
                      other=0.)
    
    ### Create an empty tensor to accumulate our dQ ###
    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    ### Load our output grad (again checking for invalid positions) ###
    dO_block = tl.load(
        dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim,
        mask=q_mask[:, None],
        other=0.
    )

    ### Load the corresponding logsumexps for this block of queries ###
    M_block = tl.load(M + offs_q, mask=q_mask, other=0.)
    M_block = M_block[:, None]
    
    ### Get the correspodning Keys and Values starting pointers ###
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim

    ### Load our precomputed D ###
    Di = tl.load(D + offs_q, mask=q_mask, other=0.)

    ### Starting kv will be 0 by default ###
    curr_kv = 0

    ### Number of steps we take is however many blocks it takes ###
    ### to cover our seq_len ###
    num_steps = tl.cdiv(SEQ_LEN, BLOCK_SIZE_KV)
    
    if STAGE == 3:
        # For causal, limit the loop to only the relevant KV blocks 
        # (up to the end of this Q block). Q cant attend to KV blocks 
        # in the future so theres no need to loop through them
        max_offs_q = start_q + BLOCK_SIZE_Q
        num_steps = tl.cdiv(max_offs_q, BLOCK_SIZE_KV)

    for _ in range(num_steps):

        ### Get our offset to the KVs we want to load ### 
        offs_kv = curr_kv + tl.arange(0, BLOCK_SIZE_KV)

        ### Mask so we dont load invalid positions ###
        kv_mask = offs_kv < SEQ_LEN

        ### Load our Data ###
        K_T_block = tl.load(kT_ptrs, mask=kv_mask[None, :], other=0.)
        V_T_block = tl.load(vT_ptrs, mask=kv_mask[None, :], other=0.)

        ### Compute our Attention Probs using teh LogSumExp Trick ###
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp2(QK_block - M_block)

        ### IF Causal we need to mask out the top triangle ###
        if STAGE == 3:
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)

        ### IF we had any invalid positions in our attention computation ###
        ### We need to zero out those probs so they dont contribute ###
        seq_mask = q_mask[:, None] & kv_mask[None, :]
        P_block = tl.where(seq_mask, P_block, 0.0)

        ### dP = dO @ v_T
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)

        ### dS = P*dP - PD = P(dP - D)
        dS_block = P_block * (dP_block - Di[:, None])

        ### Scaling for exp2 ##
        dS_block *= LN2

        ### Cast to float16 now that ops are done ###
        dS_block = dS_block.to(tl.float16)

        ### dQ = dS @ K
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        
        ### Advance our Pointers to the next chunk of KV ###
        curr_kv += BLOCK_SIZE_KV
        kT_ptrs += BLOCK_SIZE_KV * stride_seq
        vT_ptrs += BLOCK_SIZE_KV * stride_seq

    ### Store only the valid positions of our output Q ###
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block, mask=q_mask[:, None])


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

    ### This implementation assumes Seq_len is longer than 128 (the max block size of our autotune)
    ### It also assumes Seq_lens are a power of two!
    # assert SEQ_LEN >= 128, "Flash Attention supported for Context Lengths longer than 128. No real benefit to Flash Attention for short sequences"
    # assert SEQ_LEN & (SEQ_LEN - 1) == 0, "Flash attention supports only Seq Lens that are powers of 2! Max Pad if needed!" 

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
    ### We unscale this later in our backward pass! 
    softmax_scale *= 1.44269504

    O = cp.empty_like(Q)
    stage = 3 if causal else 1

    grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

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
        stride_K_seq=K.strides[2] // K.itemsize,
        stride_K_dim=K.strides[3] // K.itemsize,
        stride_V_seq=V.strides[2] // V.itemsize,
        stride_V_dim=V.strides[3] // V.itemsize,
        stride_O_seq=O.strides[2] // O.itemsize,
        stride_O_dim=O.strides[3] // O.itemsize,
        NUM_HEADS=Q.shape[1],
        SEQ_LEN=Q.shape[2],
        HEAD_DIM=HEAD_DIM_Q,
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
    dQ = cp.zeros_like(Q)
    dK = cp.zeros_like(K)
    dV = cp.zeros_like(V)

    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    
    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM**0.5

    ### Scales our log_2 for correctness, because we didnt use ln()
    softmax_scale *= 1.44269504

    # preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
    preprocess_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS)

    D = cp.empty_like(M)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

    # Compute all the elements Di
    _attn_bwd_preprocess[preprocess_grid](
        O=O.data.ptr,
        dO=dO.data.ptr,
        D=D.data.ptr,
        SEQ_LEN=SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
    )

    grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_KV"]), 1, BATCH_SIZE * NUM_HEADS)
    stage = 3 if causal else 1
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
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
    )

    grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_Q"]), 1, BATCH_SIZE * NUM_HEADS)
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
        HEAD_DIM=HEAD_DIM,
        STAGE=stage,
    )

    return dQ, dK, dV

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import time 

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
        ref_dq_cupy = cp.array(dQ_torch.detach().cpu().numpy())
        ref_dk_cupy = cp.array(dK_torch.detach().cpu().numpy())
        ref_dv_cupy = cp.array(dV_torch.detach().cpu().numpy())

        # triton implementation
        *_, tri_out, M = fused_sdpa_forward(Q, K, V, 
                                            causal=causal, 
                                            softmax_scale=softmax_scale)
        dQ, dK, dV = fused_sdpa_backward(dO, Q, K, V, tri_out, M, 
                                        softmax_scale=softmax_scale, 
                                        causal=causal)

        # compare
        tol = 1e-2
        print("Max Out Diff:", cp.abs(ref_O_cupy - tri_out).max())
        print("Max DQ Diff:", cp.abs(ref_dq_cupy - dQ).max())
        print("Max DK Diff:", cp.abs(ref_dk_cupy - dK).max())
        print("Max DV Diff:", cp.abs(ref_dv_cupy - dV).max())
        
        assert cp.abs(ref_O_cupy - tri_out).max() < tol
        assert cp.abs(ref_dq_cupy - dQ).max() < tol
        assert cp.abs(ref_dk_cupy - dK).max() < tol
        assert cp.abs(ref_dv_cupy - dV).max() < tol

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
        Q = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
        K = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
        V = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)
        dO = cp.random.normal(size=(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype)

        Q_torch = torch.tensor(Q.get(), requires_grad=True, device="cuda", dtype=torch.float16)
        K_torch = torch.tensor(K.get(), requires_grad=True, device="cuda", dtype=torch.float16)
        V_torch = torch.tensor(V.get(), requires_grad=True, device="cuda", dtype=torch.float16)
        dO_torch = torch.tensor(dO.get(), device="cuda", dtype=torch.float16)

        softmax_scale = 1.0 / (HEAD_DIM ** 0.5)
        mask = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda", dtype=torch.float16)) if causal else None

        def torch_ref_forward():
            P = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * softmax_scale
            if causal:
                P = P.masked_fill(mask == 0, float("-inf"))
            P = torch.softmax(P, dim=-1)
            O = torch.matmul(P, V_torch)
            return O

        def torch_sdpa_forward():
            return torch.nn.functional.scaled_dot_product_attention(
                Q_torch, K_torch, V_torch, attn_mask=None if not causal else mask, is_causal=causal
            )

        def triton_forward():
            # be robust in case fused_sdpa_forward returns additional items:
            ret = fused_sdpa_forward(Q, K, V, causal, softmax_scale)
            tri_out = ret[-2]  # second last is O (matching previous return signature)
            M = ret[-1]
            return tri_out, M

        def triton_backward_call(tri_out, M):
            # call backward-only kernel (expects precomputed tri_out and M)
            dQ, dK, dV = fused_sdpa_backward(dO, Q, K, V, tri_out, M, causal, softmax_scale)
            return dQ, dK, dV

        print(f"[Warmup] forward kernels (seq_len={SEQ_LEN})")
        for _ in range(warmup_iters):
            _ = torch_ref_forward()
            _ = torch_sdpa_forward()
            _ = triton_forward()
        torch.cuda.synchronize()
        cp.cuda.Device().synchronize()

        print(f"[Warmup] backward kernels (seq_len={SEQ_LEN})")
        # precompute forward outputs (we keep the computation graph for torch so backward is pure backward)
        O_ref = torch_ref_forward()         # keep graph
        O_sdpa = torch_sdpa_forward()       # keep graph
        tri_out, M = triton_forward()       # CuPy arrays

        for _ in range(warmup_iters):
            O_ref.backward(dO_torch, retain_graph=True)
            O_sdpa.backward(dO_torch, retain_graph=True)
            _ = triton_backward_call(tri_out, M)

        torch.cuda.synchronize()
        cp.cuda.Device().synchronize()


        # PyTorch naive forward timing
        start = time.time()
        for _ in range(bench_iters):
            _ = torch_ref_forward()
        torch.cuda.synchronize()
        t_ref_fwd = (time.time() - start) / bench_iters

        # PyTorch SDPA forward timing
        start = time.time()
        for _ in range(bench_iters):
            _ = torch_sdpa_forward()
        torch.cuda.synchronize()
        t_sdpa_fwd = (time.time() - start) / bench_iters

        # Triton forward timing
        start = time.time()
        for _ in range(bench_iters):
            tri_out, M = triton_forward()
        cp.cuda.Device().synchronize()
        t_tri_fwd = (time.time() - start) / bench_iters

        # Precompute forward outputs once (keep graphs) so we only measure backward work
        O_ref = torch_ref_forward()
        O_sdpa = torch_sdpa_forward()
        tri_out, M = triton_forward()  # CuPy arrays

        # PyTorch naive backward-only timing (call backward on precomputed O_ref)
        start = time.time()
        for _ in range(bench_iters):
            O_ref.backward(dO_torch, retain_graph=True)
        torch.cuda.synchronize()
        t_ref_bwd = (time.time() - start) / bench_iters

        # PyTorch SDPA backward-only timing
        start = time.time()
        for _ in range(bench_iters):
            O_sdpa.backward(dO_torch, retain_graph=True)
        torch.cuda.synchronize()
        t_sdpa_bwd = (time.time() - start) / bench_iters

        # Triton backward-only timing (call the backward kernel only)
        start = time.time()
        for _ in range(bench_iters):
            _ = fused_sdpa_backward(dO, Q, K, V, tri_out, M, causal, softmax_scale)
        cp.cuda.Device().synchronize()
        t_tri_bwd = (time.time() - start) / bench_iters

        print(f"Forward times (ms): TorchRef={t_ref_fwd*1000:.2f}, TorchSDPA={t_sdpa_fwd*1000:.2f}, Triton={t_tri_fwd*1000:.2f}")
        print(f"Backward times (ms): TorchRef={t_ref_bwd*1000:.2f}, TorchSDPA={t_sdpa_bwd*1000:.2f}, Triton={t_tri_bwd*1000:.2f}")

        return {
            "seq_len": SEQ_LEN,
            "torch_ref_fwd_ms": t_ref_fwd * 1000,
            "torch_sdpa_fwd_ms": t_sdpa_fwd * 1000,
            "triton_fwd_ms": t_tri_fwd * 1000,
            "torch_ref_bwd_ms": t_ref_bwd * 1000,
            "torch_sdpa_bwd_ms": t_sdpa_bwd * 1000,
            "triton_bwd_ms": t_tri_bwd * 1000,
        }

    print("Checking for Correctness")
    test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=128, HEAD_DIM=64, causal=False)
    test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=1, HEAD_DIM=64, causal=True)
    # test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=64, HEAD_DIM=64, causal=False)
    # test_cupy_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=64, HEAD_DIM=64, causal=False)
    print("All Correct!!")

    # print("Benchmarking")
    # all_results = []
    # seq_list = [128, 256, 512, 1024, 2048, 4096]
    # for seq_len in seq_list:
    #     print("Testing Seq Len:", seq_len)
    #     results = benchmark_flash_attention(
    #         BATCH_SIZE=4,
    #         NUM_HEADS=12,
    #         SEQ_LEN=seq_len,
    #         HEAD_DIM=64,
    #         causal=True,
    #         dtype=cp.float16,
    #         warmup_iters=5,
    #         bench_iters=50,
    #     )
    #     all_results.append(results)

    # seq_lens = np.array([r["seq_len"] for r in all_results])
    # forward_times = np.array([[r["torch_ref_fwd_ms"], r["torch_sdpa_fwd_ms"], r["triton_fwd_ms"]] for r in all_results])
    # backward_times = np.array([[r["torch_ref_bwd_ms"], r["torch_sdpa_bwd_ms"], r["triton_bwd_ms"]] for r in all_results])

    # # Forward plot
    # plt.figure(figsize=(9, 5))
    # plt.plot(seq_lens, forward_times[:, 0], "-o", label="PyTorch Naive Fwd", linewidth=2)
    # plt.plot(seq_lens, forward_times[:, 1], "-o", label="PyTorch SDPA Fwd", linewidth=2)
    # plt.plot(seq_lens, forward_times[:, 2], "-o", label="Triton FlashAttention Fwd", linewidth=2)
    # plt.xlabel("Sequence Length (L)")
    # plt.ylabel("Time (ms)")
    # plt.title("Forward Attention Runtime (forward-only)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("benchmark/flash_attn_forward.png")
    # plt.show()

    # # Backward plot
    # plt.figure(figsize=(9, 5))
    # plt.plot(seq_lens, backward_times[:, 0], "-o", label="PyTorch Naive Bwd", linewidth=2)
    # plt.plot(seq_lens, backward_times[:, 1], "-o", label="PyTorch SDPA Bwd", linewidth=2)
    # plt.plot(seq_lens, backward_times[:, 2], "-o", label="Triton FlashAttention Bwd", linewidth=2)
    # plt.xlabel("Sequence Length (L)")
    # plt.ylabel("Time (ms)")
    # plt.title("Backward Attention Runtime (backward-only)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("benchmark/flash_attn_backward.png")
    # plt.show()


from sympy import li
import torch

import triton
import triton.language as tl

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
    SEQ_LEN: tl.constexpr
):
    # 0 to the left of the diagonal (def no mask)
    if STAGE == 1:
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    # along the diagonal, some need to be masked
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    # other than non-casual attention, rest of 0 so no need to compute
    # only for non-casual attention
    else:
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # reload K and QK block
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)


        if STAGE == 2: #diagonal
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :]) # when query_index > k_index
            QK_block = QK_block + softmax_scale + tl.where(mask, 0, -1.0e6) # replace with a large negative number
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else: # don't need to mask out anything
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # effectively softmax*
        P_block = tl.math.exp(QK_block)

        # sum of rows of the attention scores (no correction yet)
        l_ij = tl.sum(P_block, 1)
        # correction factor
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
 
        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # O_block + P_block @ V_block

        # save new maximum
        m_i = m_ij

        # move to next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V[SEQ_LEN, HEAD_DIM]
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K[HEAD_DIM, SEQ_LEN]

    return O_block, l_i, m_i

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    softmax_scale,
    M, # global maximum [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
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
    BLOCK_SIZE_Q: tl.constexpr,  # aren't passed when calling since they are passed via auto tuning decorator
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    # indicates which block in the seq_len to process
    block_index_q = tl.program_id(0)

    # which head/batch to process
    index_batch_head = tl.program_id(1)
    # which batch the program is associated with
    index_batch = index_batch_head // NUM_HEADS
    # position of head in batch
    index_head = index_batch_head % NUM_HEADS

    # need smt like Q[index_batch, index_head, :, :]
    # get the (SEQ_LEN, HEAD_DIM) block
    qvk_offset  = (
        index_batch.to(tl.int64) * stride_Q_batch +
        index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base = Q + qvk_offset, # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_Q_seq, stride_Q_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0), # ensure for the right query blocks that this program works with
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        base = V + qvk_offset, # V[index_batch, index_head, :, :]
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base = K + qvk_offset, # K[index_batch, index_head, :, :]
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (stride_K_dim, stride_V_seq), # invert the strides w.r.t. Q to get QK^T
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1)
    )

    # output has same shape as query
    O_block_ptr = tl.make_block_ptr(
        base = O + qvk_offset, # K[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0)
    )

    # offsets for the tokens in the Q to process
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arrange(0, BLOCK_SIZE_Q)
    # not skipping anything, so don't need the shift
    offs_kv = tl. arrange(0, BLOCK_SIZE_KV)

    # running maximum, one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) - float("inf")
    # running sum, one for each query
    # +1.0 for log stability?
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) + 1.0
    # accumlation: for the output, a group of rows of the O matri
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype= tl.float32)

    # load the blocks of Q, staying in the SRAM
    Q_block = tl.load(Q_block_ptr)

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
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    # if STAGE == 3, need full casual attention
    # prev does the lower diagonal
    # this computes the lower diagonals of each block along the diagonal
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
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    # logsumexp for bpass
    # nice because when we subtract m_i+log(l_i) from x_i, we get
    # exp(x_i-m_i-log(l_i))=exp(x_i-m_i)/exp(log(l_i))=exp(x_i-m_i)/l_i where l_i is the normalization
    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]

    # skip by index_batch_head * SEQ_LEN based on the combined total index
    # offs_q: offsets for the tokens in Q
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q

    # store tensors for the backwards pass
    # updated maximum + log_sum
    tl.store(m_ptrs, m_i)
    # computed attention output for the current query block with the right dtype
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))

class TritonAttention(torch.autograd.Function):

    @staticmethod # no self or access to instance data, but usually used with torch.autograd.Function
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)
        grid = lambda args: {
            # cdiv: ceiling divsion -> ceil(SEQ_LEN / BLOCK_SIZE_Q) = how many blocks of Q we have
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # which group of queries are we going to worth with?
            BATCH_SIZE * NUM_HEADS, # which head of which batch element are we going to worth with?
            1 # Z in the CUDA launch grid
        }

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device = Q.device, dtype = torch.float32
        )

        # launch grid
        _attn_fwd[grid](
            Q = Q,
            K = K,
            V = V,
            softmax_scale = softmax_scale,
            M = M,
            O = O, 
            stride_Q_batch = Q.stride(0),
            stride_Q_head = Q.stride(1),
            stride_Q_seq = Q.stride(2),
            stride_Q_dim = Q.stride(3),
            stride_K_batch = K.stride(0),
            stride_K_head = K.stride(1),
            stride_K_seq = K.stride(2),
            stride_K_dim = K.stride(3),
            stride_V_batch = V.stride(0),
            stride_V_head = V.stride(1),
            stride_V_seq = V.stride(2),
            stride_V_dim = V.stride(3),
            stride_O_batch = O.stride(0),
            stride_O_head = O.stride(1),
            stride_O_seq = O.stride(2),
            stride_O_dim = O.stride(3),
            BATCH_SIZE = Q.shape[0],
            NUM_HEADS = Q.shape[1],
            SEQ_LEN = Q.shape[2],
            HEAD_DIM = HEAD_DIM_K,
            STAGE = stage
        )

        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype = torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype = dtype, device = "cuda"
        ).normal_(mean = 0.0, std = 0.5).requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype = dtype, device = "cuda"
        ).normal_(mean = 0.0, std = 0.5).requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype = dtype, device = "cuda"
        ).normal_(mean = 0.0, std = 0.5).requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM ** 0.5) # QK^T/sqrt(HEAD_DIM)
    d0 = torch.randn_like(Q) # needed for the backward pass

    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device = "cuda"))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale # swap SEQ_LEN and HEAD_DIM dimension
    if causal:
        P[:, :, MASK == 0] = float("-inf")
    P = torch.softmax(P.float(), dim = -1).half() #.half() converts back to fp16

    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0)
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare via allclose
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_0, tri_out, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dK, tri_dK, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dV, tri_dV, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol = atol, rtol = rtol)
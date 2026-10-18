"""The one tiling policy validation/ runs: every block of every validation/vector, validation/gemm and
validation/attention kernel, from the shape alone. Edit this file (or pass another with simulate_shapes.py --tiling)
and the kernels follow; GEMV blocks are PyTorchSim PYTORCHSIM's, vector blocks XLA's v6e windows, the matrix-unit
blocks a hand-picked table, attention blocks splash's and RPA's as the measurement ran them."""
BUDGET = 1 << 20
LANES = 128
PYTORCHSIM = "b08720b"
GEMV = {(384, 1152): (128, 512), (768, 768): (256, 1024), (768, 3072): (1024, 1024), (2048, 2048): (2048, 1024),
        (2048, 3072): (1024, 2048), (3072, 768): (256, 4096), (4096, 4096): (4096, 512)}


def p2(n):
    """The smallest power of two not below n."""
    return 1 << (n - 1).bit_length()


def rows_per_block(R, C, budget=BUDGET):
    """Rows of one block: the largest power of two dividing R whose block of C columns fits the budget."""
    cap = 1 << (max(1, budget // C).bit_length() - 1)
    b = 1
    while b * 2 <= cap and R % (b * 2) == 0:
        b *= 2
    return b


def rows_blocks(R, C, budget=BUDGET):
    """Rows form: each row whole, C padded to BC = p2(C) and masked, BM rows per program."""
    BC = p2(C)
    return dict(form="rows", BM=rows_per_block(R, BC, budget), BC=BC)


def split_blocks(R, C, budget=BUDGET):
    """Split form: C = A + B, A the largest power of two below C, nothing padded, BM rows of C per program."""
    A = 1 << (C.bit_length() - 1)
    return dict(form="split", BM=rows_per_block(R, C, budget), A=A, B=C - A)


def chunks_blocks(R, C, lanes=LANES):
    """Fewer rows than lanes: one [Q, R, K] block, K the smallest power of two with ceil(C / K) <= Q lanes holding
    L = C // K each; K = 1 with C = Q is the lanes form, a [Q, R] block (a unit K axis does not lower)."""
    K = 1
    while -(-C // K) > lanes:
        K *= 2
    if K == 1 and C == lanes:
        return dict(form="lanes", Q=lanes)
    return dict(form="chunks", Q=lanes, K=K, L=C // K)


def softmax_blocks(R, C, lanes=LANES, budget=BUDGET):
    """Rule set A: R = 1 is rows, R below the lanes is chunks, from there a power-of-two C is rows, any other split."""
    if R == 1:
        return rows_blocks(R, C, budget)
    if R < lanes:
        return chunks_blocks(R, C, lanes)
    return rows_blocks(R, C, budget) if C == p2(C) else split_blocks(R, C, budget)


def vector_blocks(program, R, C, lanes=LANES, budget=BUDGET):
    """The form and blocks of one shape: max and sum walk rows-form blocks in one program, slice reads one element."""
    if program == "softmax":
        return softmax_blocks(R, C, lanes, budget)
    if program in ("max", "sum"):
        return dict(rows_blocks(R, C, budget), form="walk")
    if program == "add1":
        return rows_blocks(R, C, budget)
    if program == "slice":
        return dict(form="one")
    raise KeyError(program)


def block_elements(R, C, b):
    """Elements one program's block covers, the number BUDGET bounds."""
    return {"rows": lambda: b["BM"] * b["BC"], "walk": lambda: b["BM"] * b["BC"], "split": lambda: b["BM"] * C,
            "chunks": lambda: b["Q"] * R * b["K"], "lanes": lambda: b["Q"] * R, "one": lambda: 1}[b["form"]]()


SPLASH_BLOCK = 1024
STATIC_BQ = {4096: 512, 8192: 128}
RPA_PAGE, RPA_PAGES_PER_BLOCK = 128, 32
MIN_DOT_M = 4


def attention_blocks(kind, Hq, Hkv, S):
    """prefill: splash's block min(S, 1024) queries and keys a step, or STATIC_BQ[S], the largest power of two whose
    S / BQ blocks of the static query loop, every one's scratch live, fit the double-buffer budget of 524288 B/lane
    (S 4096: 813728 at 1024; S 8192: 1674656 at 1024, 539648 at 256); decode: RPA's KV block of 32 pages of 128 tokens,
    min(S, 4096), the Hq/Hkv query heads of a group as GP rows, a power of two and at least MIN_DOT_M
    (ops/ops_dot_bf16_m2_refused.py)."""
    if kind == "prefill":
        b = STATIC_BQ.get(S, min(S, SPLASH_BLOCK))
        return dict(form="splash", BQ=b, BKV=b, GP=1)
    if kind == "decode":
        return dict(form="rpa", BQ=1, BKV=min(S, RPA_PAGE * RPA_PAGES_PER_BLOCK), GP=max(MIN_DOT_M, p2(Hq // Hkv)))
    raise KeyError(kind)


RES_B = {
    (32, 384, 1152): (32, 1024), (32, 768, 768): (32, 256), (32, 768, 3072): (32, 1024),
    (32, 2048, 2048): (32, 2048), (32, 2048, 3072): (32, 1024), (32, 3072, 768): (32, 256),
    (32, 4096, 4096): (32, 2048),
    (128, 384, 1152): (128, 1024), (128, 768, 768): (128, 256), (128, 768, 3072): (128, 1024),
    (128, 2048, 2048): (128, 2048), (128, 2048, 3072): (128, 1024), (128, 3072, 768): (128, 256),
    (128, 4096, 4096): (128, 2048),
    (256, 384, 1152): (256, 1024), (256, 768, 768): (256, 256), (256, 768, 3072): (256, 1024),
    (256, 2048, 2048): (256, 2048), (256, 2048, 3072): (256, 1024), (256, 3072, 768): (256, 256),
    (512, 384, 1152): (512, 1024), (512, 768, 768): (512, 256), (512, 768, 3072): (512, 1024),
    (512, 2048, 2048): (512, 1024), (512, 2048, 3072): (512, 1024), (512, 3072, 768): (512, 256),
    (512, 4096, 4096): (256, 2048),
    (1024, 384, 1152): (1024, 1024), (1024, 768, 3072): (1024, 1024), (1024, 2048, 2048): (1024, 512),
    (1024, 2048, 3072): (1024, 512), (1024, 4096, 4096): (256, 2048),
    (2048, 768, 3072): (2048, 1024), (2048, 2048, 3072): (2048, 512), (2048, 4096, 4096): (256, 2048),
}
RES_A = {
    (8192, 2048, 2048): (512, 2048), (8192, 2048, 3072): (512, 2048),
    (1024, 768, 768): (256, 1024), (1024, 3072, 768): (256, 512), (2048, 384, 1152): (128, 1024),
    (2048, 768, 768): (256, 1024), (2048, 2048, 2048): (2048, 512), (2048, 3072, 768): (256, 2048),
    (4096, 384, 1152): (128, 2048), (4096, 768, 768): (256, 2048), (4096, 768, 3072): (1024, 512),
    (4096, 2048, 2048): (2048, 512), (4096, 2048, 3072): (512, 2048), (4096, 3072, 768): (256, 2048),
    (4096, 4096, 4096): (512, 1024), (8192, 384, 1152): (128, 2048), (8192, 768, 768): (256, 2048),
    (8192, 768, 3072): (1024, 512), (8192, 3072, 768): (256, 2048),
}
TILED = {
    (8192, 4096, 4096): (2048, 2048, 1024, "grid"),
}


def pow2_pieces(n, cap=None):
    """n as power-of-two pieces, largest first, none above cap (greedy; with no cap the binary digits of n)."""
    out = []
    while n:
        p = 1 << (n.bit_length() - 1)
        p = p if cap is None else min(p, cap)
        out.append(p)
        n -= p
    return out


def gemv_blocks(K, N):
    """(K, N) -> XBLOCK, R0_BLOCK as PyTorchSim PYTORCHSIM pins them for torch.matmul [1,K]@[K,N] bf16 on tpuv6e,
    read from the fixed_config of the kernels it generated (validation/gemm shapes)."""
    X, R = GEMV[K, N]
    return dict(XBLOCK=X, R0_BLOCK=R)


def gemm_blocks(M, K, N, form):
    """The blocks of one shape in `form`: K padded to p2(K) and masked when Kwhole, or in binary pieces when K
    overflows whole; W/cap (B resident), Wn/Mc (A resident) and BM/BN/BK/template (tiled) from the tables."""
    if form == "gemv":
        return gemv_blocks(K, N)
    if form == "res_b(Kwhole)":
        W, cap = RES_B[M, K, N]
        return dict(Kp=p2(K), W=W, n_pieces=pow2_pieces(N, cap))
    if form == "res_a(Kwhole)":
        Wn, Mc = RES_A[M, K, N]
        return dict(Kp=p2(K), Wn=Wn, Mc=Mc)
    if form.startswith("res_a(Kpieces"):
        Wn, Mc = RES_A[M, K, N]
        return dict(k_pieces=pow2_pieces(K), Wn=Wn, Mc=Mc)
    if form == "tiled_fallback":
        BM, BN, BK, template = TILED[M, K, N]
        return dict(BM=BM, BN=BN, BK=BK, template=template)
    raise KeyError(form)

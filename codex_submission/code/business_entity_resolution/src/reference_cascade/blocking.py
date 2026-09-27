"""Sparse IDF-weighted token blocking (candidate retrieval).

For every country pool (Source 2 + Source 3 records of that country) we build an inverted index
over these tokens (exact vocabulary, 64-bit hashes, no collisions that matter):

    address unigrams        'a' namespace  (tokens of addr_norm)
    address bigrams         'b'            (adjacent addr_norm tokens: '113 back', 'back bay' ...)
    name unigrams           'n'            (tokens of name_core)
    glued name              'n'            (name_alnum for multi-token names, so 'wilford hancock'
                                            and 'wilfordhancock.com' share a token)

Tokens are weighted by IDF (df measured on the pool), records are L2-normalised, and the score of a
(S1, pool) pair is the cosine of those sparse vectors == sum of shared IDF^2 / norms.
Tokens whose document frequency exceeds `max_df` are dropped from retrieval (they cost a lot and
carry almost no identity information). Each S1 keeps its top-K pool records.

Country is a hard key: pools are built per country, never crossed. Country is an open string label,
so France (absent from train) needs no special handling.
"""
from dataclasses import asdict, dataclass, field
import multiprocessing as mp
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

_A, _B, _N, _X, _Y, _P, _S = 11, 22, 33, 44, 55, 66, 77          # hash seeds == namespaces


@dataclass
class View:
    """One independent retrieval pass. use_name=False -> address-only vectors (a pool record's own,
    unmatchable name tokens then cannot dilute its address evidence)."""
    name: str = "A"
    use_name: bool = True
    power: float = 1.0            # score = shared_weight / (||q|| * ||p||**power)
    top_k: int = 10               # per (S1, source)


@dataclass
class BlockingConfig:
    max_df: int = 3000            # drop tokens more frequent than this in the pool
    addr_bigrams: bool = True
    name_glued: bool = True
    name_weight: float = 1.0      # multiplier on name-token weights vs address-token weights
    num_deletions: bool = True    # SymSpell 1-deletion keys for numeric tokens (digit drop / typo tolerant)
    name_deletions: bool = False  # tested: hurts recall (noise) -> off
    del_weight: float = 0.6
    pair_keys: bool = True        # (number token, other address token) composite keys: rare even when
                                  # every single token is common (e.g. '10, HOWRAH, West Bengal')
    pair_weight: float = 1.0
    max_nums: int = 3             # numeric tokens per record used for pair keys
    num_sets: bool = False        # order-free key of the WHOLE set of numeric tokens (+ every 1-removed
                                  # subset): robust to component reordering / one missing or wrong number
    set_weight: float = 1.0
    views: list = field(default_factory=lambda: [View("A", True, 1.0, 10), View("B", False, 1.0, 8)])
    min_score: float = 0.0
    chunk: int = 4000             # queries per spgemm chunk

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["views"] = [View(**v) for v in d.get("views", [])]
        return cls(**d)


# ------------------------------------------------------------------------------ tokenisation
def _explode(strs: pl.Series):
    df = (pl.DataFrame({"s": strs}).with_row_index("r")
          .with_columns(pl.col("s").str.split(" ")).explode("s"))
    return df.filter(pl.col("s").is_not_null() & (pl.col("s") != ""))


def _deletion_keys(tok: pl.DataFrame, minlen: int, maxlen: int) -> pl.DataFrame:
    """SymSpell: identity + every single-character deletion of tokens with minlen<=len<=maxlen.
    Two tokens within edit distance 1 share at least one key."""
    ln = tok["s"].str.len_chars()
    d = tok.filter((ln >= minlen) & (ln <= maxlen))
    outs = [d.select("r", "s")]
    dl = d["s"].str.len_chars()
    for i in range(maxlen):
        di = d.filter(dl > i)
        if di.height and i < maxlen:
            outs.append(pl.DataFrame({"r": di["r"], "s": di["s"].str.slice(0, i) + di["s"].str.slice(i + 1)}))
    return pl.concat(outs)


def token_entries(df: pl.DataFrame, cfg: BlockingConfig):
    """-> (row u32[], hash u64[], weight f32[], is_name bool[]) : one entry per (record, token)."""
    rs, hs, ws, gs = [], [], [], []

    def add(r, h, w, name=False):
        rs.append(np.asarray(r, dtype=np.uint32)); hs.append(np.asarray(h, dtype=np.uint64))
        ws.append(np.full(len(r), w, dtype=np.float32)); gs.append(np.full(len(r), name, dtype=bool))

    a = _explode(df["addr_norm"])
    add(a["r"].to_numpy(), a["s"].hash(seed=_A).to_numpy(), 1.0)
    if cfg.addr_bigrams:
        nxt = a["s"].shift(-1)
        m = (a["r"] == a["r"].shift(-1)).fill_null(False)
        b = pl.DataFrame({"r": a["r"], "s": a["s"] + "_" + nxt}).filter(m)
        add(b["r"].to_numpy(), b["s"].hash(seed=_B).to_numpy(), 1.0)
    if cfg.pair_keys:
        isnum = a["s"].str.contains(r"\d")
        nums = a.filter(isnum).with_columns(pl.int_range(pl.len()).over("r").alias("k")).filter(pl.col("k") < cfg.max_nums)
        oth = a.filter(~isnum).select("r", pl.col("s").alias("o"))
        pk = nums.select("r", "s").join(oth, on="r")
        add(pk["r"].to_numpy(), (pk["s"] + "|" + pk["o"]).hash(seed=_P).to_numpy(), cfg.pair_weight)
        if cfg.max_nums > 1:                       # number-number pairs (unordered adjacency-free)
            n2 = nums.select("r", pl.col("s").alias("s2"), "k")
            nn = nums.select("r", "s", pl.col("k").alias("k1")).join(n2, on="r").filter(pl.col("k1") < pl.col("k"))
            add(nn["r"].to_numpy(), (nn["s"] + "|" + nn["s2"]).hash(seed=_P).to_numpy(), cfg.pair_weight)
    if cfg.num_sets:
        nt = a.filter(pl.col("s").str.contains(r"\d"))
        if nt.height:
            rr = nt["r"].to_numpy(); hh = nt["s"].hash(seed=_S).to_numpy()
            starts = np.flatnonzero(np.concatenate([[True], rr[1:] != rr[:-1]]))
            cnt = np.diff(np.concatenate([starts, [len(rr)]]))
            total = np.add.reduceat(hh, starts)                 # uint64 wrap-around sum == order-free set hash
            per_e_total = np.repeat(total, cnt); per_e_cnt = np.repeat(cnt, cnt)
            full = (cnt >= 2) & (cnt <= 8)
            add(rr[starts][full], total[full] * np.uint64(0x9E3779B97F4A7C15), cfg.set_weight)
            sub = (per_e_cnt >= 3) & (per_e_cnt <= 8)                 # every (n-1)-subset
            add(rr[sub], (per_e_total[sub] - hh[sub]) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(1), cfg.set_weight)
    if cfg.num_deletions:
        nt = a.filter(pl.col("s").str.contains(r"\d"))
        k = _deletion_keys(nt, 3, 7)
        add(k["r"].to_numpy(), k["s"].hash(seed=_X).to_numpy(), cfg.del_weight)
    n = _explode(df["name_core"])
    add(n["r"].to_numpy(), n["s"].hash(seed=_N).to_numpy(), cfg.name_weight, True)
    if cfg.name_deletions:
        k = _deletion_keys(n.filter(~pl.col("s").str.contains(r"\d")), 5, 12)
        add(k["r"].to_numpy(), k["s"].hash(seed=_Y).to_numpy(), cfg.del_weight * cfg.name_weight, True)
    if cfg.name_glued:
        multi = df["name_core"].str.contains(" ")
        g = pl.DataFrame({"r": np.arange(df.height, dtype=np.uint32), "s": df["name_alnum"]}).filter(multi)
        add(g["r"].to_numpy(), g["s"].hash(seed=_N).to_numpy(), cfg.name_weight, True)
    return np.concatenate(rs), np.concatenate(hs), np.concatenate(ws), np.concatenate(gs)


# ------------------------------------------------------------------------------ index
class PoolIndex:
    """Inverted index of one pool (one country, one source): token -> weighted postings, one CSR per view."""

    def __init__(self, pool: pl.DataFrame, cfg: BlockingConfig, chunk_rows: int = 1_000_000):
        self.cfg, self.n = cfg, pool.height
        R, H, W, G = [], [], [], []
        for off in range(0, pool.height, chunk_rows):
            r, h, w, g = token_entries(pool.slice(off, chunk_rows), cfg)
            R.append(r + np.uint32(off)); H.append(h); W.append(w); G.append(g)
        r, h, w, g = np.concatenate(R), np.concatenate(H), np.concatenate(W), np.concatenate(G)
        del R, H, W, G
        order = np.argsort(h, kind="stable")
        h, r, w, g = h[order], r[order], w[order], g[order]
        del order
        vocab, start, cnt = np.unique(h, return_index=True, return_counts=True)
        keep_tok = cnt <= cfg.max_df
        idf = np.log((self.n + 1.0) / (cnt + 1.0)).astype(np.float32)
        tok = np.repeat(np.arange(len(vocab), dtype=np.int32), cnt)
        del h
        keep_e = keep_tok[tok]
        r, w, g, tok = r[keep_e], w[keep_e], g[keep_e], tok[keep_e]
        wt = idf[tok] * w
        self.vocab, self.idf, self.df = vocab, idf * keep_tok, cnt
        self.pt, self.inv_norm = [], []
        for v in cfg.views:
            m = slice(None) if v.use_name else ~g
            rv, tv, wv = r[m], tok[m], wt[m]
            norm = np.sqrt(np.bincount(rv, weights=wv.astype(np.float64) ** 2, minlength=self.n)).astype(np.float32)
            norm[norm == 0] = 1.0
            c = np.bincount(tv, minlength=len(vocab))
            indptr = np.concatenate([[0], np.cumsum(c)]).astype(np.int64)
            self.pt.append(sp.csr_matrix((wv, rv.astype(np.int32), indptr), shape=(len(vocab), self.n)))
            self.inv_norm.append((norm ** -v.power).astype(np.float32))

    def query(self, q: pl.DataFrame):
        """One sparse query matrix per view (n_q x V): weights idf*ns_w / ||q_view||."""
        r, h, w, g = token_entries(q, self.cfg)
        pos = np.searchsorted(self.vocab, h)
        pos[pos >= len(self.vocab)] = 0
        ok = (self.vocab[pos] == h) & (self.idf[pos] > 0)
        r, pos, w, g = r[ok], pos[ok], w[ok], g[ok]
        wt = self.idf[pos] * w
        out = []
        for v in self.cfg.views:
            m = np.ones(len(r), bool) if v.use_name else ~g
            rv, pv, wv = r[m], pos[m], wt[m]
            norm = np.sqrt(np.bincount(rv, weights=wv.astype(np.float64) ** 2, minlength=q.height)).astype(np.float32)
            norm[norm == 0] = 1.0
            out.append(sp.csr_matrix((wv / norm[rv], (rv.astype(np.int32), pv)), shape=(q.height, len(self.vocab))))
        return out


# ------------------------------------------------------------------------------ retrieval
def _topk_rows(S: sp.csr_matrix, k: int, min_score: float):
    """per-row top-k of a CSR matrix -> (row, col, val) arrays, best first."""
    indptr, indices, data = S.indptr, S.indices, S.data
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = indptr[i], indptr[i + 1]
        if a == b:
            continue
        d = data[a:b]
        sel = np.argpartition(d, -k)[-k:] if b - a > k else np.arange(b - a)
        sel = sel[d[sel] >= min_score]
        sel = sel[np.argsort(-d[sel], kind="stable")]
        rows.append(np.full(len(sel), i, dtype=np.int32)); cols.append(indices[a:b][sel]); vals.append(d[sel])
    if not rows:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


_G = {}


def _work(args):
    off, n = args
    idx, Qs, cfg = _G["idx"], _G["Q"], _G["cfg"]
    out = []
    for vi, v in enumerate(cfg.views):
        S = Qs[vi][off:off + n] @ idx.pt[vi]
        S.data *= idx.inv_norm[vi][S.indices]
        r, c, s = _topk_rows(S, v.top_k, cfg.min_score)
        out.append((r + off, c, s))
    return out


def retrieve(idx: PoolIndex, q: pl.DataFrame, cfg: BlockingConfig, workers: int = 8):
    """Top-K pool rows for every query row and every view.
    Returns a list (one entry per view) of (q_row, pool_row, score), each sorted by q_row, -score.

    Sparse query matrices are built here, in the parent: forked workers must not touch polars
    (its thread pool deadlocks after fork) - they only do scipy/numpy."""
    Qs = idx.query(q)
    _G.update(idx=idx, Q=Qs, cfg=cfg)
    jobs = [(o, min(cfg.chunk, q.height - o)) for o in range(0, q.height, cfg.chunk)]
    if workers <= 1 or len(jobs) == 1:
        res = [_work(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as p:
            res = p.map(_work, jobs, chunksize=1)
    _G.clear()
    return [tuple(np.concatenate([x[vi][k] for x in res]) for k in range(3)) for vi in range(len(cfg.views))]

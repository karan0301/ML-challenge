"""Pair features for (S1, candidate) pairs.

All features are language / country agnostic (no country one-hot): France is unseen in training, so the
model must only see *similarities*, never country-specific identities.

Groups
  name     : exact flags, rapidfuzz ratios (Levenshtein / token-set / token-sort / partial / Jaro-Winkler),
             token Jaccard + containment, IDF-weighted token cosine, lengths
  address  : same families on addr_norm + numeric-token features (house / unit number equality)
  retrieval: blocking score, rank, ratio to the best candidate, gap to next
  context  : how generic the name / address is (frequency of identical name among S1 and pool)
Missing pool addresses are NaN (not 0): "no evidence" is different from "evidence of mismatch".
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import _explode
from .normalization import LEGAL_TOKENS

_NAME_SEED, _ADDR_SEED, _NUM_SEED = 101, 102, 103


# ----------------------------------------------------------------------------- df tables
class DFTable:
    """token-hash -> document frequency over the pool of one country (used for IDF weights)."""

    def __init__(self, h: np.ndarray, cnt: np.ndarray, n_docs: int):
        o = np.argsort(h)
        self.h, self.cnt, self.n = h[o], cnt[o], n_docs

    def idf(self, q: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.h, q)
        pos[pos >= len(self.h)] = 0
        df = np.where(self.h[pos] == q, self.cnt[pos], 0)
        return np.log((self.n + 1.0) / (df + 1.0)).astype(np.float32)


def build_df_tables(pool: pl.DataFrame, chunk=1_000_000):
    """{ (country, field): DFTable } from the concatenated S2+S3 pool of a split."""
    out = {}
    countries = pool["country"].cast(pl.Utf8).unique().to_list()
    for c in countries:
        p = pool.filter(pl.col("country").cast(pl.Utf8) == c)
        for field, seed in (("name_core", _NAME_SEED), ("addr_norm", _ADDR_SEED)):
            parts = []
            for off in range(0, p.height, chunk):
                e = _explode(p[field].slice(off, chunk))
                parts.append(pl.DataFrame({"h": e["s"].hash(seed=seed)}).group_by("h").len())
            g = pl.concat(parts).group_by("h").agg(pl.col("len").sum())
            out[(c, field)] = DFTable(g["h"].to_numpy(), g["len"].to_numpy(), p.height)
    return out


# ----------------------------------------------------------------------------- helpers
def _cd(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _rowdot(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def pair_token_stats(a: pl.Series, b: pl.Series, countries: pl.Series, seed: int, tables: dict, field: str):
    """a[i], b[i] strings; returns dict of float32 arrays: inter, na, nb, jacc, cont_a, cont_b, cos (idf)."""
    n = len(a)
    ea, eb = _explode(a), _explode(b)
    ra, ha = ea["r"].to_numpy(), ea["s"].hash(seed=seed).to_numpy()
    rb, hb = eb["r"].to_numpy(), eb["s"].hash(seed=seed).to_numpy()
    cs = countries.cast(pl.Utf8).to_numpy()
    idf_a = np.zeros(len(ha), np.float32); idf_b = np.zeros(len(hb), np.float32)
    for c in np.unique(cs):
        t = tables[(c, field)]
        ma, mb = cs[ra] == c, cs[rb] == c
        idf_a[ma] = t.idf(ha[ma]); idf_b[mb] = t.idf(hb[mb])
    allh, inv = np.unique(np.concatenate([ha, hb]), return_inverse=True)
    V = len(allh)
    A = sp.csr_matrix((np.ones(len(ha), np.float32), (ra, inv[:len(ha)])), shape=(n, V))
    B = sp.csr_matrix((np.ones(len(hb), np.float32), (rb, inv[len(ha):])), shape=(n, V))
    A.data[:] = 1; B.data[:] = 1                       # binary (duplicates summed above)
    inter = _rowdot(A, B)
    na = np.asarray(A.sum(1)).ravel().astype(np.float32); nb = np.asarray(B.sum(1)).ravel().astype(np.float32)
    Aw = sp.csr_matrix((idf_a, (ra, inv[:len(ha)])), shape=(n, V)); Bw = sp.csr_matrix((idf_b, (rb, inv[len(ha):])), shape=(n, V))
    Aw.data = np.minimum(Aw.data, 50); Bw.data = np.minimum(Bw.data, 50)
    winter = _rowdot(Aw, Bw)
    wa = np.sqrt(np.asarray(Aw.multiply(Aw).sum(1)).ravel()); wb = np.sqrt(np.asarray(Bw.multiply(Bw).sum(1)).ravel())
    cos = winter / np.maximum(wa * wb, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = inter / (na + nb - inter)
        ca = inter / na
        cb = inter / nb
    return dict(inter=inter, na=na, nb=nb, jacc=jacc, cont_a=ca, cont_b=cb, cos=cos.astype(np.float32))


# ----------------------------------------------------------------------------- record-level derived columns
_LEGAL = pl.Series(sorted(t for t in LEGAL_TOKENS if t not in ("com", "www")))
_SK_CONS = "bdfgjklmnpqrstv"
_LEGAL_RE = r"\b(?:" + "|".join(sorted(_LEGAL.to_list(), key=len, reverse=True)) + r")\b"


def skeleton(name_alnum: pl.Series) -> pl.Series:
    """Consonant skeleton: crude phonetic key that survives vowel typos, doubled letters and (roughly)
    Devanagari/Telugu/Tamil -> Latin transliteration ('classic industries' ~ 'klaasik indsttriij')."""
    s = name_alnum
    for a, b in (("ph", "f"), ("ck", "k"), ("sh", "s"), ("q", "k"), ("c", "k"), ("x", "ks"), ("z", "s"), ("w", "v")):
        s = s.str.replace_all(a, b, literal=True)
    s = s.str.replace_all("[aeiouyh]", "")
    for ch in _SK_CONS + "s":
        s = s.str.replace_all(f"{ch}{ch}+", ch)
    return s


def add_record_columns(df: pl.DataFrame) -> pl.DataFrame:
    legal = df["name_full"].str.extract_all(_LEGAL_RE).list.join(" ").fill_null("")
    return df.with_columns(legal.alias("name_legal"), skeleton(df["name_alnum"]).alias("name_skel"))


# ----------------------------------------------------------------------------- context (frequency) columns
def add_context_columns(s1: pl.DataFrame, pools: dict) -> tuple:
    """Record-level derived columns + genericness (how many records share this exact name / address)."""
    s1 = add_record_columns(s1)
    pools = {k: add_record_columns(v) for k, v in pools.items()}
    s1 = s1.with_columns(
        pl.len().over(["country", "name_core"]).cast(pl.UInt32).alias("name_freq"),
        pl.when(pl.col("addr_norm") == "").then(None).otherwise(pl.len().over(["country", "addr_norm"]))
        .cast(pl.UInt32).alias("addr_freq"))
    out = {}
    for src, p in pools.items():
        out[src] = p.with_columns(
            pl.len().over(["country", "name_core"]).cast(pl.UInt32).alias("name_freq"),
            pl.when(pl.col("addr_norm") == "").then(None).otherwise(pl.len().over(["country", "addr_norm"]))
            .cast(pl.UInt32).alias("addr_freq"))
    return s1, out


# ----------------------------------------------------------------------------- main
NAME_COLS = ["name_full", "name_core", "name_alnum", "name_sorted", "name_legal", "name_skel"]
ADDR_COLS = ["addr_norm", "addr_nums"]


def pair_features(cand: pl.DataFrame, s1: pl.DataFrame, pools: dict, tables: dict) -> pl.DataFrame:
    """cand: (s1_row, src, pool_row, score, rank[, ...]). s1/pools must contain the *_freq columns
    (see add_context_columns) and all NAME_COLS + ADDR_COLS. Returns a feature frame aligned with cand."""
    n = cand.height
    s1_idx = cand["s1_row"].to_numpy(); src = cand["src"].to_numpy(); prow = cand["pool_row"].to_numpy()

    # ---- gather pool-side columns in candidate order
    cols = NAME_COLS + ADDR_COLS + ["name_freq", "addr_freq", "country"]
    P = {c: None for c in cols}
    for c in cols:
        parts = np.empty(n, dtype=object) if c not in ("name_freq", "addr_freq") else np.zeros(n, np.float32)
        for s in (2, 3):
            m = src == s
            if not m.any():
                continue
            g = pools[s][c].gather(prow[m])
            if c in ("name_freq", "addr_freq"):
                parts[m] = g.cast(pl.Float32).fill_null(np.nan).to_numpy()
            elif c == "country":
                parts[m] = g.cast(pl.Utf8).to_numpy()
            else:
                parts[m] = g.to_numpy()
        P[c] = parts
    Q = {c: s1[c].gather(s1_idx) for c in NAME_COLS + ADDR_COLS + ["name_freq", "addr_freq", "country"]}

    def L(x):
        return x.to_list() if isinstance(x, pl.Series) else list(x)

    F = {}
    # ---------------- name
    q_core, p_core = L(Q["name_core"]), list(P["name_core"])
    q_full, p_full = L(Q["name_full"]), list(P["name_full"])
    q_alnum, p_alnum = L(Q["name_alnum"]), list(P["name_alnum"])
    q_sort, p_sort = L(Q["name_sorted"]), list(P["name_sorted"])
    F["n_exact_full"] = np.array([x == y for x, y in zip(q_full, p_full)], np.float32)
    F["n_exact_core"] = np.array([x == y for x, y in zip(q_core, p_core)], np.float32)
    F["n_exact_alnum"] = np.array([x == y for x, y in zip(q_alnum, p_alnum)], np.float32)
    F["n_exact_sorted"] = np.array([x == y for x, y in zip(q_sort, p_sort)], np.float32)
    F["n_ratio_full"] = _cd(q_full, p_full, fuzz.ratio) / 100
    F["n_ratio_core"] = _cd(q_core, p_core, fuzz.ratio) / 100
    F["n_ratio_alnum"] = _cd(q_alnum, p_alnum, fuzz.ratio) / 100
    F["n_tsr"] = _cd(q_core, p_core, fuzz.token_set_ratio) / 100
    F["n_tsort"] = _cd(q_core, p_core, fuzz.token_sort_ratio) / 100
    F["n_partial"] = _cd(q_alnum, p_alnum, fuzz.partial_ratio) / 100
    F["n_jw"] = _cd(q_core, p_core, JaroWinkler.normalized_similarity)
    F["n_lev"] = _cd(q_alnum, p_alnum, Levenshtein.normalized_similarity)
    st = pair_token_stats(Q["name_core"], pl.Series(p_core), Q["country"], _NAME_SEED, tables, "name_core")
    F["n_jacc"], F["n_cont1"], F["n_cont2"], F["n_cos"] = st["jacc"], st["cont_a"], st["cont_b"], st["cos"]
    F["n_inter"], F["n_tok1"], F["n_tok2"] = st["inter"], st["na"], st["nb"]
    q_sk, p_sk = L(Q["name_skel"]), list(P["name_skel"])
    F["n_skel_ratio"] = _cd(q_sk, p_sk, fuzz.ratio) / 100
    F["n_skel_exact"] = np.array([x == y for x, y in zip(q_sk, p_sk)], np.float32)
    F["n_skel_len"] = np.minimum(pl.Series(q_sk).str.len_chars().to_numpy(), pl.Series(p_sk).str.len_chars().to_numpy()).astype(np.float32)
    F["n_lev_abs"] = _cd(q_alnum, p_alnum, Levenshtein.distance)
    q_lg, p_lg = L(Q["name_legal"]), list(P["name_legal"])
    F["legal_eq"] = np.array([x == y for x, y in zip(q_lg, p_lg)], np.float32)
    F["legal_1_empty"] = np.array([x == "" for x in q_lg], np.float32)
    F["legal_2_empty"] = np.array([y == "" for y in p_lg], np.float32)
    F["n_len1"] = Q["name_core"].str.len_chars().to_numpy().astype(np.float32)
    F["n_len2"] = pl.Series(p_core).str.len_chars().to_numpy().astype(np.float32)

    # ---------------- address
    q_a, p_a = L(Q["addr_norm"]), list(P["addr_norm"])
    p_empty = np.array([x == "" for x in p_a])
    F["a_p_empty"] = p_empty.astype(np.float32)
    ar = lambda x: np.where(p_empty, np.nan, x).astype(np.float32)
    F["a_exact"] = ar(np.array([x == y for x, y in zip(q_a, p_a)], np.float32))
    F["a_ratio"] = ar(_cd(q_a, p_a, fuzz.ratio) / 100)
    F["a_tsr"] = ar(_cd(q_a, p_a, fuzz.token_set_ratio) / 100)
    F["a_tsort"] = ar(_cd(q_a, p_a, fuzz.token_sort_ratio) / 100)
    F["a_partial"] = ar(_cd(q_a, p_a, fuzz.partial_ratio) / 100)
    F["a_jw"] = ar(_cd(q_a, p_a, JaroWinkler.normalized_similarity))
    sa = pair_token_stats(Q["addr_norm"], pl.Series(p_a), Q["country"], _ADDR_SEED, tables, "addr_norm")
    F["a_jacc"], F["a_cont1"], F["a_cont2"], F["a_cos"] = ar(sa["jacc"]), ar(sa["cont_a"]), ar(sa["cont_b"]), ar(sa["cos"])
    F["a_inter"], F["a_tok1"], F["a_tok2"] = ar(sa["inter"]), sa["na"], sa["nb"]
    F["a_len1"] = Q["addr_norm"].str.len_chars().to_numpy().astype(np.float32)
    F["a_len2"] = pl.Series(p_a).str.len_chars().to_numpy().astype(np.float32)
    # numeric tokens (house / unit / PIN numbers)
    q_n, p_n = Q["addr_nums"], pl.Series(list(P["addr_nums"]))
    qs = [set(x.split()) for x in L(q_n)]; ps = [set(x.split()) for x in p_n.to_list()]
    ni = np.array([len(x & y) for x, y in zip(qs, ps)], np.float32)
    nq = np.array([len(x) for x in qs], np.float32); npp = np.array([len(y) for y in ps], np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["num_jacc"] = ar(ni / (nq + npp - ni))
        F["num_cont1"] = ar(ni / nq)
    F["num_inter"] = ar(ni); F["num_n1"] = nq; F["num_n2"] = ar(npp)
    q_first = Q["addr_nums"].str.split(" ").list.first().fill_null("").to_list()
    p_first = [x.split(" ", 1)[0] for x in P["addr_nums"]]
    F["a_lev_abs"] = ar(_cd(q_a, p_a, Levenshtein.distance))
    F["num_lev"] = ar(_cd(L(q_n), list(P["addr_nums"]), Levenshtein.distance))
    F["num_first_lev"] = ar(_cd(q_first, p_first, Levenshtein.distance))
    F["num_first_eq"] = ar(np.array([(a != "" and a == b) for a, b in zip(q_first, p_first)], np.float32))
    # first numeric token of S1 appears anywhere among the candidate's numbers
    F["num_first_in"] = ar(np.array([(a != "" and a in y) for a, y in zip(q_first, ps)], np.float32))

    # ---------------- retrieval / context
    views = sorted(int(c[2:]) for c in cand.columns if c.startswith("rk"))
    for vi in views:
        sc_i = cand[f"sc{vi}"].fill_null(np.nan).to_numpy().astype(np.float32)
        F[f"r{vi}_score"] = sc_i
        F[f"r{vi}_rank"] = cand[f"rk{vi}"].cast(pl.Float32).fill_null(np.nan).to_numpy()
        top = cand.select(pl.col(f"sc{vi}").max().over(["s1_row", "src"]).alias("t"),
                          pl.col(f"sc{vi}").max().over("s1_row").alias("t_all"))
        F[f"r{vi}_ratio_top"] = (sc_i / top["t"].fill_null(np.nan).to_numpy()).astype(np.float32)
        F[f"r{vi}_ratio_all"] = (sc_i / top["t_all"].fill_null(np.nan).to_numpy()).astype(np.float32)
    F["r_rank"] = cand["rank"].to_numpy().astype(np.float32)
    F["r_n_views"] = sum(np.isfinite(F[f"r{vi}_rank"]) for vi in views).astype(np.float32)
    F["r_n_all"] = cand.select(pl.len().over("s1_row"))["len"].to_numpy().astype(np.float32)
    F["src3"] = (src == 3).astype(np.float32)
    F["s1_name_freq"] = Q["name_freq"].cast(pl.Float32).to_numpy()
    F["p_name_freq"] = P["name_freq"]
    F["s1_addr_freq"] = Q["addr_freq"].cast(pl.Float32).fill_null(np.nan).to_numpy()
    F["p_addr_freq"] = P["addr_freq"]
    # combos
    F["both_exact_core"] = F["n_exact_core"] * np.nan_to_num(F["a_exact"])
    F["name_x_addr"] = F["n_tsr"] * np.nan_to_num(F["a_tsr"])
    F["max_na"] = np.maximum(F["n_tsr"], np.nan_to_num(F["a_tsr"]))
    a_tsr_filled = np.where(np.isnan(F["a_tsr"]), F["n_tsr"], F["a_tsr"])   # no address evidence -> fall back to name
    F["min_na"] = np.minimum(F["n_tsr"], a_tsr_filled)
    F["addr_strong_name_weak"] = np.nan_to_num(F["a_ratio"]) * (1 - F["n_tsr"])    # obfuscated-name / address-anchored matches
    F["name_strong_addr_weak"] = F["n_tsr"] * (1 - a_tsr_filled)
    F["num_strong_name_weak"] = np.nan_to_num(F["num_jacc"]) * (1 - F["n_tsr"])
    return pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})


def chunk_bounds(cand: pl.DataFrame, chunk: int):
    """Row boundaries of ~`chunk` pairs that never split one S1's candidates (per-S1 rank features
    stay exact). cand must be sorted by s1_row."""
    s = cand["s1_row"].to_numpy()
    starts = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    bounds, last = [0], 0
    for x in starts:
        if x - last >= chunk:
            bounds.append(int(x)); last = int(x)
    bounds.append(cand.height)
    return bounds


def feature_chunks(cand, s1, pools, tables, chunk=400_000, verbose=True):
    """Generator of (start, end, feature_frame) - lets inference stream predictions without ever
    holding the full (~35M x 58) feature matrix."""
    import time
    b = chunk_bounds(cand, chunk)
    t = time.time()
    for i in range(len(b) - 1):
        yield b[i], b[i + 1], pair_features(cand[b[i]:b[i + 1]], s1, pools, tables)
        if verbose:
            print(f"    features {b[i + 1]:,}/{cand.height:,}  {time.time() - t:.0f}s", flush=True)


def featurise_chunked(cand, s1, pools, tables, chunk=400_000, verbose=True):
    return pl.concat([f for _, _, f in feature_chunks(cand, s1, pools, tables, chunk, verbose)])

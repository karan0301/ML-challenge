"""Name / address normalisation.

Nothing here looks anything up externally: the tables below are static
abbreviation dictionaries (street types, legal forms, US/Indian state names).

Every field is kept in several representations so no information is destroyed:

  name_full    canonical tokens, legal forms kept                  'rentz inc services 66460'
  name_core    legal forms / '.com' removed                        'rentz services 66460'
  name_alnum   name_core with spaces removed (catches 'domain' names, word gluing)
  name_sorted  sorted unique tokens of name_core (word-order transposition)
  addr_norm    canonical *short* forms (st, rd, ave, blvd, r ...) tokens in original order
  addr_nums    space-joined tokens that contain a digit (house / unit / PIN numbers)
  addr_alnum   addr_norm with spaces removed

Canonical forms are the SHORT ones: 'street','st','saint' -> 'st', 'road','rd' -> 'rd', 'rue','r' -> 'r'.
Mapping several long forms onto one short token is harmless (the mapping is applied identically to
every source) and makes the pipeline country-agnostic: it works for France, which is not in train.

Speed: rows are ~20M per split, so everything is polars expressions (Aho-Corasick replace_many on
double-space-padded text = word-boundary-safe multi-pattern replacement). unidecode (transliteration
of Devanagari / Telugu / Tamil / accents) is only run on the non-ASCII subset.
"""
import re

import polars as pl
from unidecode import unidecode

# ----------------------------------------------------------------------------- dictionaries
# canonical short form -> variants (all lower-case, tokens already stripped of punctuation)
_ADDR_MAP = {
    "st": ["street", "str", "saint", "sainte", "ste"],
    "rd": ["road"],
    "ave": ["avenue", "av", "aven"],
    "blvd": ["boulevard", "bd", "bvd", "boul", "blv"],
    "dr": ["drive"],
    "ln": ["lane"],
    "ct": ["court"],
    "cir": ["circle"],
    "hwy": ["highway"],
    "pkwy": ["parkway", "pky"],
    "pl": ["place"],
    "ter": ["terrace"],
    "sq": ["square"],
    "trl": ["trail"],
    "aly": ["alley"],
    "rte": ["route", "rt"],
    "expy": ["expressway"],
    "r": ["rue"],
    "all": ["allee", "alle"],
    "imp": ["impasse"],
    "chem": ["chemin"],
    "crs": ["cours"],
    "n": ["north"], "s": ["south"], "e": ["east"], "w": ["west"],
    "flr": ["floor"],
    "apt": ["apartment", "apartments", "appartement", "appt"],
    "bldg": ["building", "batiment", "bat"],
    "opp": ["opposite"],
    "no": ["number", "num", "nr"],
    "cmplx": ["complex"],
    "rdg": ["ridge"],
    "hts": ["heights"],
    "mt": ["mount"],
    "ft": ["fort"],
    "pt": ["point"],
    "sec": ["sector"],
    "colony": ["colny"],
}

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}
_IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk",
    "chandigarh": "ch", "puducherry": "py", "pondicherry": "py", "ladakh": "ld",
}
# 'new york' is both state and city; mapping is applied to every source identically so this is safe.

# legal-form canonical (kept in name_full, dropped in name_core)
_LEGAL_MAP = {
    "corp": ["corporation", "corpn"],
    "inc": ["incorporated"],
    "ltd": ["limited"],
    "pvt": ["private"],
    "co": ["company"],
    "llc": [], "llp": [], "lp": [], "plc": [], "pte": [],
    "sarl": [], "sas": [], "sasu": [], "sci": [], "eurl": [], "sa": [], "snc": [], "sca": [],
}
LEGAL_TOKENS = set(_LEGAL_MAP) | {v for vs in _LEGAL_MAP.values() for v in vs} | {"com", "www"}

_ACRONYMS = {  # dotted acronyms -> glued (applied before punctuation removal)
    "l.l.p.": "llp", "l.l.p": "llp", "l.l.c.": "llc", "l.l.c": "llc", "s.a.r.l.": "sarl",
    "s.a.r.l": "sarl", "s.a.s.u.": "sasu", "s.a.s.u": "sasu", "s.a.s.": "sas", "s.a.s": "sas",
    "s.c.i.": "sci", "s.c.i": "sci", "e.u.r.l.": "eurl", "e.u.r.l": "eurl", "s.a.": "sa",
    "p.v.t.": "pvt", "p.v.t": "pvt", "l.t.d.": "ltd", "l.t.d": "ltd", "u.s.a.": "usa",
}


def _pad(words: str) -> str:
    """' foo  bar '  : single padding space, tokens separated by two spaces."""
    return " " + "  ".join(words.split()) + " "


def _build(pairs):
    """pairs of (variant, canonical) -> (patterns, replacements), longest pattern first
    so multi-word variants ('north carolina') win over their single-word parts."""
    seen, out = set(), []
    for v, c in pairs:
        if v not in seen and v != c:
            seen.add(v); out.append((_pad(v), _pad(c)))
    out.sort(key=lambda x: -len(x[0]))
    return [p for p, _ in out], [r for _, r in out]


def _pairs(mapping):
    return [(v, canon) for canon, variants in mapping.items() for v in variants]


# state full-name -> code (US and India share a few codes, e.g. 'ar', 'mp'; that is fine: the
# mapping keys are the full names, which are distinct)
_ADDR_PATS, _ADDR_REPS = _build(_pairs(_ADDR_MAP) + list(_US_STATES.items()) + list(_IN_STATES.items()))
_LEGAL_PATS, _LEGAL_REPS = _build(_pairs(_LEGAL_MAP))

# ----------------------------------------------------------------------------- transliteration
_RUN = re.compile(r"[^\x00-\x7F]+")
_INDIC = re.compile(r"[ऀ-෿]")


def _tr_run(m):
    r = m.group()
    t = unidecode(r)
    if _INDIC.search(r):
        t = t.replace("N", "")          # unidecode emits capital N for anusvara / nasal marks
    return t


def transliterate(s: pl.Series) -> pl.Series:
    """unidecode only the non-ASCII runs of only the non-ASCII rows."""
    idx = s.str.contains(r"[^\x00-\x7F]").arg_true()
    if len(idx) == 0:
        return s
    vals = s.gather(idx).to_list()
    return s.scatter(idx, pl.Series([_RUN.sub(_tr_run, v) for v in vals], dtype=pl.Utf8))


def _basic(s: pl.Series, keep_dots: bool) -> pl.Series:
    """raw -> lower ascii; punctuation -> spaces. Returns padded double-space text."""
    s = s.str.replace_all("[°º˚ª]", " ").str.replace_all(r"(?i)<null>", " ")
    s = transliterate(s).str.to_lowercase()
    s = s.str.replace_all("[’'`´]", "").str.replace_all("&", " and ")
    return s


def _finish(s: pl.Series) -> pl.Series:
    return s.str.replace_all(r"\s+", " ").str.strip_chars()


def _tokens_padded(s: pl.Series) -> pl.Series:
    """'a b c' -> '  a  b  c  ' so replace_many is token-boundary safe."""
    return ("  " + s.str.replace_all(" ", "  ") + "  ")


# ----------------------------------------------------------------------------- public API
def normalise_names(name: pl.Series) -> pl.DataFrame:
    s = _basic(name, keep_dots=True)
    # dotted acronyms and .com suffix, before dots become spaces
    s = s.str.replace_many(list(_ACRONYMS), list(_ACRONYMS.values()))
    s = s.str.replace_all(r"\.(com|net|org)\b", " com ")
    s = s.str.replace_all(r"[^a-z0-9]+", " ")
    s = _finish(s)
    s = _tokens_padded(s).str.replace_many(_LEGAL_PATS, _LEGAL_REPS)
    full = _finish(s)
    # core: drop legal-form tokens
    legal = pl.Series(sorted(LEGAL_TOKENS))
    core = _finish(full.str.split(" ").list.eval(
        pl.element().filter(~pl.element().is_in(legal.implode()))).list.join(" "))
    core = pl.select(pl.when(core == "").then(full).otherwise(core)).to_series()
    return pl.DataFrame({
        "name_full": full,
        "name_core": core,
        "name_alnum": core.str.replace_all(" ", ""),
        "name_sorted": core.str.split(" ").list.unique(maintain_order=False).list.sort().list.join(" "),
    })


_ORD_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
              "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
              "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
              "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
              "twentieth": "20"}
_ORD_PATS, _ORD_REPS = _build(list(_ORD_WORDS.items()))


def canonical_numbers(a: pl.Series) -> pl.Series:
    """'4th','4nd','10rd','fourth' -> '4','4','10','4'; '001427' -> '1427'. Input: addr_norm tokens."""
    a = _tokens_padded(a).str.replace_many(_ORD_PATS, _ORD_REPS)
    a = a.str.replace_all(r"\b(\d+)(?:st|nd|rd|th)\b", "${1}")
    a = a.str.replace_all(r"\b0+(\d)", "${1}")
    return _finish(a)


def normalise_addresses(addr: pl.Series) -> pl.DataFrame:
    s = _basic(addr, keep_dots=False)
    s = s.str.replace_all(r"[^a-z0-9]+", " ")
    s = _finish(s)
    s = _tokens_padded(s).str.replace_many(_ADDR_PATS, _ADDR_REPS)
    s = s.str.replace_all("  null  ", "  ")
    a = canonical_numbers(_finish(s))
    nums = a.str.extract_all(r"\b\w*\d\w*\b").list.join(" ")
    return pl.DataFrame({
        "addr_norm": a,
        "addr_nums": nums.fill_null(""),
        "addr_alnum": a.str.replace_all(" ", ""),
    })


def normalise_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Raw source frame -> raw columns + all normalised representations."""
    n = normalise_names(df["business_name"])
    a = normalise_addresses(df["business_address"])
    return pl.concat([df.select("entity_id", "business_name", "business_address", "country"), n, a],
                     how="horizontal")

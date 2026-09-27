"""External-memory, multi-pass candidate generation using DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _normal_sql(column: str) -> str:
    # DuckDB's strip_accents keeps the blocking keys useful for French while the
    # model applies a fuller Python NFKD normalization before scoring.
    return (
        "trim(regexp_replace(regexp_replace(lower(strip_accents(coalesce("
        f"{column}, ''))), '&', ' and ', 'g'), '[^[:alnum:]]+', ' ', 'g'))"
    )


def build_candidates(data_dir: Path, split: str, work_dir: Path, token_df_limit: int = 120) -> Path:
    """Create a DuckDB database containing the final model candidate set.

    It makes complementary exact-name, exact-address, and rare-token passes.
    Rare tokens are calculated within each open-set country label, so no country
    list is encoded in the pipeline.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "duckdb_tmp").mkdir(parents=True, exist_ok=True)
    db_path = work_dir / f"{split}_blocking.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute("SET memory_limit='11GB'")
    con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{_sql_path(work_dir / 'duckdb_tmp')}'")
    con.execute("SET preserve_insertion_order=false")

    prefix = "train" if split == "train" else "test"
    s1_file = data_dir / split / f"{prefix}_source1.tsv"
    s2_file = data_dir / split / f"{prefix}_source2.tsv"
    s3_file = data_dir / split / f"{prefix}_source3.tsv"
    source_select = (
        f"SELECT entity_id, country, business_name, business_address, "
        f"{_normal_sql('business_name')} AS name_norm, "
        f"{_normal_sql('business_address')} AS address_norm "
        f"FROM read_csv('{_sql_path(s1_file)}', delim='\\t', header=true, all_varchar=true)"
    )
    target_select = (
        " UNION ALL ".join(
            f"SELECT entity_id, country, business_name, business_address, "
            f"{_normal_sql('business_name')} AS name_norm, "
            f"{_normal_sql('business_address')} AS address_norm "
            f"FROM read_csv('{_sql_path(path)}', delim='\\t', header=true, all_varchar=true)"
            for path in (s2_file, s3_file)
        )
    )
    con.execute(f"CREATE TABLE source1 AS {source_select}")
    con.execute(f"CREATE TABLE targets AS {target_select}")
    con.execute("CREATE INDEX source1_id_idx ON source1(entity_id)")
    con.execute("CREATE INDEX target_id_idx ON targets(entity_id)")

    con.execute(
        "CREATE TABLE target_tokens AS "
        "SELECT DISTINCT country, entity_id, token FROM targets, "
        "UNNEST(string_split(name_norm || ' ' || address_norm, ' ')) AS u(token) "
        "WHERE length(token) >= 3"
    )
    con.execute(
        "CREATE TABLE source_tokens AS "
        "SELECT DISTINCT country, entity_id, token FROM source1, "
        "UNNEST(string_split(name_norm || ' ' || address_norm, ' ')) AS u(token) "
        "WHERE length(token) >= 3"
    )
    con.execute(
        f"CREATE TABLE rare_tokens AS SELECT country, token FROM target_tokens "
        f"GROUP BY country, token HAVING count(*) <= {int(token_df_limit)}"
    )
    con.execute("CREATE INDEX target_tokens_idx ON target_tokens(country, token)")
    con.execute("CREATE INDEX source_tokens_idx ON source_tokens(country, token)")
    con.execute(
        "CREATE TABLE token_pairs AS "
        "SELECT s.entity_id AS source1_entity_id, t.entity_id AS candidate_entity_id, "
        "count(DISTINCT s.token) AS rare_shared_tokens, sum(1.0 / sqrt(tc.df)) AS rare_score "
        "FROM source_tokens s JOIN rare_tokens r ON s.country=r.country AND s.token=r.token "
        "JOIN target_tokens t ON r.country=t.country AND r.token=t.token "
        "JOIN (SELECT country, token, count(*) AS df FROM target_tokens GROUP BY country, token) tc "
        "ON r.country=tc.country AND r.token=tc.token "
        "GROUP BY s.entity_id, t.entity_id"
    )
    con.execute(
        "CREATE TABLE candidates AS "
        "SELECT source1_entity_id, candidate_entity_id, max(exact_name) AS exact_name_block, "
        "max(exact_address) AS exact_address_block, max(rare_shared_tokens) AS rare_shared_tokens, "
        "max(rare_score) AS rare_score FROM ("
        "SELECT s.entity_id AS source1_entity_id, t.entity_id AS candidate_entity_id, 1 AS exact_name, 0 AS exact_address, 0 AS rare_shared_tokens, 0.0 AS rare_score "
        "FROM source1 s JOIN targets t ON s.country=t.country AND s.name_norm=t.name_norm WHERE s.name_norm <> '' "
        "UNION ALL SELECT s.entity_id, t.entity_id, 0, 1, 0, 0.0 FROM source1 s JOIN targets t "
        "ON s.country=t.country AND s.address_norm=t.address_norm WHERE s.address_norm <> '' "
        "UNION ALL SELECT source1_entity_id, candidate_entity_id, 0, 0, rare_shared_tokens, rare_score FROM token_pairs"
        ") GROUP BY source1_entity_id, candidate_entity_id"
    )
    con.execute("CREATE INDEX candidates_s1_idx ON candidates(source1_entity_id)")
    con.execute("ANALYZE")
    con.close()
    return db_path

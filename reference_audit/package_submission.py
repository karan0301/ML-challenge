"""Build <team>_submission.zip in the layout required by the challenge:

    <team>_submission.zip
      output/{matching_results.tsv, candidate_pairs.tsv}
      code/business_entity_resolution/{src/, README.md, requirements.txt, experiments/, artifacts/models, artifacts/metrics}
      Documentation_template.md          (the filled-in methodology document)

    python package_submission.py --team MyTeam
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--team", default="TEAM")
ap.add_argument("--doc", default=str(ROOT / "Documentation_template.md"))
a = ap.parse_args()

out = ROOT.parent / f"{a.team}_submission.zip"
code = "code/business_entity_resolution"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(ROOT / "output" / f, f"output/{f}")
    for p in sorted((ROOT / "src").glob("*.py")):
        z.write(p, f"{code}/src/{p.name}")
    for p in ("README.md", "requirements.txt"):
        z.write(ROOT / p, f"{code}/{p}")
    for p in sorted((ROOT / "experiments").glob("*")):
        if p.suffix in (".py", ".csv"):
            z.write(p, f"{code}/experiments/{p.name}")
    for sub in ("models", "metrics"):
        for p in sorted((ROOT / "artifacts" / sub).glob("*")):
            z.write(p, f"{code}/artifacts/{sub}/{p.name}")
    z.write(a.doc, "Documentation_template.md")
print(f"wrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")
with zipfile.ZipFile(out) as z:
    for n in z.namelist():
        print("  ", n)

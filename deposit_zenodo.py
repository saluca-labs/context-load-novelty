"""Deposit the context-load novelty paper to Zenodo (saluca community).
Mirrors the proven deposit_infocosmo.py flow.

  python deposit_zenodo.py            # create/refresh DRAFT (upload PDF + metadata), STOP
  python deposit_zenodo.py --publish  # publish existing draft -> mints DOI (permanent)
"""
import sys, json
from pathlib import Path
import requests

ENV = Path(r"C:\AI\daily-brief\.env")
HERE = Path(__file__).resolve().parent
PDF = HERE / "paper.pdf"
STATE = HERE / ".zenodo_draft.json"
BASE = "https://zenodo.org/api"


def getv(name):
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


TOKEN = getv("ZENODO_API_TOKEN")
if not TOKEN:
    sys.exit("no ZENODO_API_TOKEN in .env")
H = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
HA = {"Authorization": f"Bearer {TOKEN}"}

DESC = (
 "<p>Does the amount of relevant context in a language model's window act as a knob on the "
 "novelty of its output, and is there an optimal input-to-window ratio? Using a controlled "
 "ideation task with embedding-based novelty and multi-judge groundedness metrics across four "
 "models (gemma2:9b through a frontier model), we find: (1) relevant context injected as neutral "
 "reference &mdash; with no instruction to be original &mdash; reliably moves ideation off the "
 "mainstream answer, replicating across every family (+49% to +147%), with an isolation design "
 "ruling out the implicit-instruction confound; (2) the effect is a <em>saturating switch, not a "
 "tunable dial</em> &mdash; there is no optimal ratio, novelty saturating within a few hundred "
 "tokens of relevant context; and (3) this absolute threshold is scale-invariant, if anything "
 "earlier on the stronger model (~300 vs ~800 tokens).</p>"
 "<p>A three-judge groundedness protocol (fast LLM, strong LLM, then a blinded human tiebreaker) "
 "finds groundedness flat across load (no coherence cost) and, as a second contribution, that "
 "automated LLM groundedness judges correlate near zero with a human rater and systematically "
 "over-rate on speculative ideation &mdash; manufacturing a spurious high-load 'erosion' that only "
 "the human pass caught. A cautionary result for LLM-as-judge evaluation.</p>"
 "<p><strong>Scope:</strong> all evaluation rests on automated proxies plus a single generalist "
 "human rater; authoritative judgment of the ideas' quality, originality, and field-specific "
 "soundness would require domain experts in each area (a sleep scientist; product and business "
 "practitioners), flagged as the essential next step. Our metrics establish the load-response "
 "shape, not expert-graded creative value.</p>"
 "<p><strong>Authorship:</strong> Cristian Ruvalcaba and the Saluca Agentic AI Research Team "
 "(Saluca LLC). Code and data: https://github.com/saluca-labs/context-load-novelty . "
 "Not peer-reviewed.</p>"
)

META = {"metadata": {
    "title": "Relevant Context-Load as a Saturating Novelty Knob in Language Models: A Cross-Family Study",
    "upload_type": "publication",
    "publication_type": "preprint",
    "description": DESC,
    "creators": [
        {"name": "Ruvalcaba, Cristian"},
        {"name": "Saluca Agentic AI Research Team", "affiliation": "Saluca LLC"},
    ],
    "keywords": ["large language models", "in-context learning", "context window", "creativity",
                 "novelty", "ideation", "LLM-as-judge", "prompt engineering", "empirical evaluation"],
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": "1",
    "communities": [{"identifier": "saluca"}],
    "notes": ("Authored by Cristian Ruvalcaba with the Saluca agentic AI research team (a multi-agent "
              "AI system). The human researcher originated the question and hypothesis, made all "
              "methodological and go/no-go decisions, performed the blinded human rating, reviewed the "
              "outputs, and is accountable for the claims; the agents built the harness, ran the "
              "experiments, analysed the data, and drafted the manuscript under human direction. "
              "Results derive from local execution and released data. Not peer-reviewed."),
}}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(d):
    STATE.write_text(json.dumps(d, indent=1))


def main():
    publish = "--publish" in sys.argv[1:]
    st = load_state()
    if publish:
        dep_id = st.get("deposit_id")
        if not dep_id:
            sys.exit("no existing draft to publish (run without --publish first)")
        r = requests.post(f"{BASE}/deposit/depositions/{dep_id}/actions/publish", headers=H, timeout=90)
        r.raise_for_status()
        j = r.json()
        doi = j.get("doi") or j.get("metadata", {}).get("doi")
        st.update({"doi": doi, "html": j["links"].get("html"), "published": True})
        save_state(st)
        print("PUBLISHED  DOI:", doi)
        print("URL:", j["links"].get("html"))
        return

    dep_id = st.get("deposit_id")
    if dep_id:
        r = requests.get(f"{BASE}/deposit/depositions/{dep_id}", headers=H, timeout=90)
        if r.status_code != 200:
            dep_id = None
    if not dep_id:
        r = requests.post(f"{BASE}/deposit/depositions", headers=H, json={}, timeout=90)
        r.raise_for_status()
        dep = r.json(); dep_id = dep["id"]
        st["deposit_id"] = dep_id; st["bucket"] = dep["links"]["bucket"]
        save_state(st)
        print("draft created:", dep_id)
    bucket = st["bucket"]

    fr = requests.get(f"{BASE}/deposit/depositions/{dep_id}/files", headers=H, timeout=90); fr.raise_for_status()
    for f in fr.json():
        requests.delete(f"{BASE}/deposit/depositions/{dep_id}/files/{f['id']}", headers=H, timeout=90)
    with PDF.open("rb") as fh:
        ru = requests.put(f"{bucket}/{PDF.name}", headers=HA, data=fh.read(), timeout=180)
    ru.raise_for_status()
    print(f"uploaded {PDF.name} ({PDF.stat().st_size//1024} KB)")

    rm = requests.put(f"{BASE}/deposit/depositions/{dep_id}", headers=H, json=META, timeout=90)
    rm.raise_for_status()
    dep = rm.json()
    st["html"] = dep["links"].get("html")
    save_state(st)
    print("metadata applied. DRAFT NOT PUBLISHED.")
    print("deposit_id:", dep_id)
    print("review URL:", dep["links"].get("html"))
    pd = dep.get("metadata", {}).get("prereserve_doi", {})
    print("reserved DOI (on publish):", pd.get("doi") if pd else "n/a")


if __name__ == "__main__":
    main()

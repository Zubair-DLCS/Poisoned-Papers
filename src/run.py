"""
Run the experiment grid for one model.

  python src/run.py --provider ollama --model qwen2.5:7b
  python src/run.py --provider groq --model <model> --per-class 10

For every defence: screen each sampled record as published (clean), then
  * every include-attack on the records the review EXCLUDED (smuggling a paper in), and
  * every exclude-attack on the records the review INCLUDED (knocking a paper out).
Decisions go to results/decisions_<model>.csv; raw replies are cached, so an interrupted run
resumes where it stopped. Decisions already in that file are kept and not asked again, so
`--attacks neutral_out` adds one condition to a finished run without repeating the rest:

  python src/run.py --provider ollama --model qwen2.5:7b --attacks neutral_out
"""
import os, argparse, random, re, time
import pandas as pd
from common import RES, SEED
from data import sample, criteria
from attacks import INCLUDE_ATTACKS, EXCLUDE_ATTACKS, inject, goal
from agent import DEFENCES, screen
from llm import make_llm


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def jobs(records, crit, defences, only=None):
    for defence in defences:
        for _, r in records.iterrows():
            topic = crit[r.review]["topic"]
            if not only or "clean" in only:
                yield defence, r, "clean", r.abstract
            attacks = INCLUDE_ATTACKS if r.included == 0 else EXCLUDE_ATTACKS
            for a in attacks:
                if not only or a in only:
                    yield defence, r, a, inject(r.abstract, a, topic, variant=int(r.record_id[-3:]))


def run(provider=None, model=None, per_class=10, defences=DEFENCES, limit=None, res_dir=RES,
        attacks=None):
    os.makedirs(res_dir, exist_ok=True)
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    model = model or os.environ.get("LLM_MODEL", "gullible")
    tag = slug(f"{provider}:{model}")
    llm = make_llm(provider, model, cache_path=os.path.join(res_dir, f"cache_{tag}.jsonl"))
    records, crit = sample(per_class), criteria()
    out = os.path.join(res_dir, f"decisions_{tag}.csv")
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    done = set(zip(old.defence, old.record_id, old.attack)) if len(old) else set()
    rows, t0 = [], time.time()
    all_jobs = [j for j in jobs(records, crit, defences, attacks)
                if (j[0], j[1].record_id, j[2]) not in done][:limit]
    print(f"{len(done)} decisions already in {os.path.basename(out)}; {len(all_jobs)} to run", flush=True)
    for i, (defence, r, attack, abstract) in enumerate(all_jobs, 1):
        rng = random.Random(f"{SEED}-{r.record_id}-{attack}")
        decision, reply = screen(llm, crit[r.review], r.title, abstract, defence, rng)
        rows.append(dict(model=llm.name, record_id=r.record_id, review=r.review,
                         included=r.included, defence=defence, attack=attack,
                         goal=goal(attack) if attack != "clean" else "", decision=decision,
                         reply=reply.replace("\n", " ")[:300]))
        if i % 50 == 0 or i == len(all_jobs):
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
            print(f"{i}/{len(all_jobs)} calls, {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.concat([old, pd.DataFrame(rows)], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--per-class", type=int, default=10,
                    help="included and excluded records sampled per review (4 reviews)")
    ap.add_argument("--defences", nargs="+", default=DEFENCES)
    ap.add_argument("--limit", type=int, help="stop after this many calls (smoke test)")
    ap.add_argument("--attacks", nargs="+", help="run only these conditions (e.g. neutral_out)")
    a = ap.parse_args()
    run(a.provider, a.model, a.per_class, a.defences, a.limit, attacks=a.attacks)

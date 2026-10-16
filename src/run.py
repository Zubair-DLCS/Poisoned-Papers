"""
Run the experiment grid for one model.

  python src/run.py --provider ollama --model qwen2.5:7b
  python src/run.py --provider groq --model <model> --per-class 10

For every defence: screen each sampled record as published (clean), then
  * every include-attack on the records the review EXCLUDED (smuggling a paper in), and
  * every exclude-attack on the records the review INCLUDED (knocking a paper out).
Decisions go to results/decisions_<model>.csv; raw replies are cached, so an interrupted run
resumes where it stopped.
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


def jobs(records, crit, defences):
    for defence in defences:
        for _, r in records.iterrows():
            topic = crit[r.review]["topic"]
            yield defence, r, "clean", r.abstract
            attacks = INCLUDE_ATTACKS if r.included == 0 else EXCLUDE_ATTACKS
            for a in attacks:
                yield defence, r, a, inject(r.abstract, a, topic, variant=int(r.record_id[-3:]))


def run(provider=None, model=None, per_class=10, defences=DEFENCES, limit=None, res_dir=RES):
    os.makedirs(res_dir, exist_ok=True)
    provider = provider or os.environ.get("LLM_PROVIDER", "mock")
    model = model or os.environ.get("LLM_MODEL", "gullible")
    tag = slug(f"{provider}:{model}")
    llm = make_llm(provider, model, cache_path=os.path.join(res_dir, f"cache_{tag}.jsonl"))
    records, crit = sample(per_class), criteria()
    rows, t0 = [], time.time()
    all_jobs = list(jobs(records, crit, defences))[:limit]
    for i, (defence, r, attack, abstract) in enumerate(all_jobs, 1):
        rng = random.Random(f"{SEED}-{r.record_id}-{attack}")
        decision, reply = screen(llm, crit[r.review], r.title, abstract, defence, rng)
        rows.append(dict(model=llm.name, record_id=r.record_id, review=r.review,
                         included=r.included, defence=defence, attack=attack,
                         goal=goal(attack) if attack != "clean" else "", decision=decision,
                         reply=reply.replace("\n", " ")[:300]))
        if i % 50 == 0 or i == len(all_jobs):
            pd.DataFrame(rows).to_csv(os.path.join(res_dir, f"decisions_{tag}.csv"), index=False)
            print(f"{i}/{len(all_jobs)} calls, {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider"); ap.add_argument("--model")
    ap.add_argument("--per-class", type=int, default=10,
                    help="included and excluded records sampled per review (4 reviews)")
    ap.add_argument("--defences", nargs="+", default=DEFENCES)
    ap.add_argument("--limit", type=int, help="stop after this many calls (smoke test)")
    a = ap.parse_args()
    run(a.provider, a.model, a.per_class, a.defences, a.limit)

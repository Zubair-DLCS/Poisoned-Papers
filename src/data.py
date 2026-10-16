"""
Real screening data from four software-engineering systematic reviews (titles, abstracts and
the reviewers' final include/exclude labels), released with FASTREAD (Yu, Kraft and Menzies,
2018; MIT licence). Downloaded from GitHub into data/raw/.

A balanced sample is drawn from each review (default: 25 included and 25 excluded records),
so attack success can be measured on both kinds of paper with a modest number of model calls.
"""
import os, json, re, subprocess
import pandas as pd
from common import RAW, DATA, SEED

REPO = "https://github.com/fastread/src.git"
REVIEWS = ["Hall", "Wahono", "Radjenovic", "Kitchenham"]


def fetch():
    dest = os.path.join(RAW, "fastread")
    if not os.path.isdir(dest):
        os.makedirs(RAW, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, dest], check=True)
    return dest


def clean(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t


def load_review(name):
    d = pd.read_csv(os.path.join(fetch(), "workspace", "data", f"{name}.csv"), encoding="latin-1")
    abstract_col = "Abstract" if "Abstract" in d.columns else "abs"
    out = pd.DataFrame({"review": name, "title": d["Document Title"].map(clean),
                        "abstract": d[abstract_col].map(clean),
                        "included": (d["label"].str.lower() == "yes").astype(int)})
    out = out[(out.abstract.str.len() > 200) & (out.title.str.len() > 0)]
    return out.drop_duplicates("title").reset_index(drop=True)


def criteria():
    return json.load(open(os.path.join(DATA, "reviews.json"), encoding="utf8"))


def sample(per_class=25, seed=SEED):
    parts = []
    for name in REVIEWS:
        d = load_review(name)
        for lab in (1, 0):
            parts.append(d[d.included == lab].sample(n=per_class, random_state=seed))
    s = pd.concat(parts, ignore_index=True)
    s.insert(0, "record_id", [f"{r}-{i:03d}" for i, r in enumerate(s.review)])
    return s


if __name__ == "__main__":
    for n in REVIEWS:
        d = load_review(n)
        print(n, len(d), "included", d.included.sum())
    print(sample().groupby(["review", "included"]).size())

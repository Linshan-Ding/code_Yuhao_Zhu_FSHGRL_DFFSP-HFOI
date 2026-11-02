#!/usr/bin/env python3
"""Prose-style checks for academic text (items 14-27 of the manuscript checker), standalone.

    python stylecheck.py sections/*.tex                  # LaTeX sections
    python stylecheck.py paper/zh-draft.tex --zh         # Chinese text (auto-detected when >30% CJK)
    python stylecheck.py draft.md notes.txt              # Markdown / plain text (**bold** counts as emphasis)
    python stylecheck.py sections/*.tex --baseline style_baseline.json
    python stylecheck.py sections/*.tex --method-name FeasiSched   # the one CamelCase name that is allowed

Items
 14. no semicolons survive in the running prose
 15. sentence-initial connectives stay within budget per section
 16. none of the banned stock phrases appears
 17. hedging stays within budget and never stacks
 18. coined names and self-made acronyms stay within the naming budget
 19. no mid-sentence colon in the running prose
 20. no dash used as a parenthetical in the running prose
 21. no bold or italic emphasis inside a sentence; run-in headings within budget
 22. the abstract is a single paragraph
 23. sentence length varies; triplet lists stay within budget
 24. intensifiers, generic verbs and abstract nouns stay within density limits
 25. every prose paragraph in method/experiment sections carries something concrete
 26. no citation cluster of more than four keys
 27. a reverse outline (first sentence of every paragraph) is written for review
 30. obscure vocabulary where the field uses a plain word
 31. symbols and hyphens: arrows and operators standing in for words (prose and tables),
     hyphen chains, prefix hyphens the field does not use, CamelCase coined names
 32. proofs and derivations: no paragraph-style proofs, no inline relation chains,
     multi-line derivations carry step justifications or labels
 33. scaffolding: enumeration labels, summary and roadmap paragraphs, parenthetical
     density, "Note that"-style openers (folded into item 15), lists in running text

Items 28-29 belong to the manuscript checker (submission package, anonymity), so the
style items continue at 30. The numbering continues the manuscript checker's items 1-13 (consistency and venue
limits, in check.py of the opt-paper-manuscript skill), which imports run_style_checks
from this file. A line that legitimately breaks a rule carries a `% style-exempt`
comment. Thresholds come from style_baseline.json when present (style_baseline.py
measures them from published papers in the target venue) and from the built-in
defaults otherwise. Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


# ---- 14-27: prose style ---------------------------------------------------
# Written drafts, not the revision pass, are where style is won: these checks
# run after every section is drafted, then once more over the whole manuscript.

MATH_ENVS = r"(?:equation|align|gather|multline|eqnarray|algorithmic|tabular|tabularx|thebibliography|center|titlepage)\*?"
# Macros whose argument is structural text (captions, placeholders, notes), not running prose.
STRUCTURAL_MACROS = {"caption": 1, "figplace": 4, "draftnote": 1, "label": 1, "includegraphics": 1,
                     "input": 1, "bibliography": 1, "cite": 1, "citep": 1, "citet": 1,
                     "ref": 1, "eqref": 1, "texttt": 1, "url": 1}


def strip_macro(t: str, name: str, nargs: int) -> str:
    """Remove \\name{...}{...} with balanced braces, keeping newlines for line numbers."""
    out, i, pat = [], 0, re.compile(r"\\" + name + r"\*?(?:\[[^\]]*\])?\s*\{")
    while True:
        m = pat.search(t, i)
        if not m:
            out.append(t[i:]); break
        out.append(t[i:m.start()])
        j, removed = m.end() - 1, 0
        while removed < nargs and j < len(t) and t[j] == "{":
            depth, k = 0, j
            while k < len(t):
                depth += t[k] == "{"
                depth -= t[k] == "}"
                k += 1
                if depth == 0:
                    break
            out.append("\n" * t[j:k].count("\n")); j, removed = k, removed + 1
            while j < len(t) and t[j] in " \t":
                j += 1
        out.append(" "); i = j
    return "".join(out)


def prose_of(text: str) -> str:
    """Strip everything that legitimately carries semicolons, colons, bold or keywords.

    Multi-line strips keep their newlines so line numbers still match the
    source file (the style-exempt lookup depends on that)."""
    keep_lines = lambda m: "\n" * m.group(0).count("\n")
    t = strip_comments(text)
    if "\\begin{document}" in t:                              # single-file sources: drop the preamble
        i = t.index("\\begin{document}")
        t = "\n" * t[:i].count("\n") + t[i:]
    t = re.sub(r"\\(?:re)?newcommand\*?\{[^}]*\}(?:\[\d\])*\{.*?\}\s*$", keep_lines, t, flags=re.M | re.S)
    t = re.sub(r"^.*(?:\\noindent\s*\\textbf\{(?:关键词|Highlights|Keywords)|通讯作者|Corresponding author)[^\n]*$", "", t, flags=re.M)
    t = re.sub(r"〔[^〕]*〕", " ", t)                              # template placeholders
    t = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % MATH_ENVS, keep_lines, t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", keep_lines, t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\\[.*?\\\]", keep_lines, t, flags=re.S)
    for name, n in STRUCTURAL_MACROS.items():
        t = strip_macro(t, name, n)
    t = re.sub(r"\\item\[[^\]]*\]", r"\\item", t)          # list labels such as \item[\textbf{A1}]
    t = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", t)
    return t


# ---- word lists: English -------------------------------------------------
CONNECTIVES = ("moreover", "furthermore", "additionally", "in addition", "notably",
               "consequently", "specifically", "overall", "meanwhile", "nevertheless", "thus",
               "note that", "importantly", "interestingly", "crucially", "it is important to note",
               "in particular", "in other words", "that is,", "to this end", "in this way")
BANNED = ("it is worth noting", "plays a crucial role", "plays a pivotal role",
          "delve into", "leverage", "in this section, we will", "aims to")
HEDGES = ("may ", "might ", "could potentially", "to some extent", "relatively ", "arguably")
INTENSIFIERS = ("significantly", "substantially", "remarkably", "effectively", "seamlessly",
                "comprehensively", "robustly", "dramatically", "greatly", "highly")
GENERIC_VERBS = ("facilitate", "enable", "ensure", "enhance", "demonstrate", "exhibit", "achieve", "utilize")
ABSTRACT_NOUNS = ("framework", "paradigm", "mechanism", "scheme", "module", "pipeline", "strategy")
NAME_ADJ = ("efficient", "robust", "novel", "smart", "adaptive", "dynamic", "hierarchical", "intelligent")
SUFFIX = r"(?:aware|driven|guided|enhanced|adaptive|based|oriented)"
COMMON_ACRONYMS = {"PPO", "GNN", "GAT", "GCN", "DRL", "RL", "VRP", "TSP", "JSSP", "FJSP", "CVRP", "MDP",
                   "MLP", "CNN", "RNN", "LSTM", "LLM", "GPU", "CPU", "SOTA", "MIP", "MILP", "CP", "GA",
                   "SA", "PSO", "TS", "LKH", "HV", "IGD", "OOD", "API", "CI", "SGD", "KL", "GAE",
                   "AM", "POMO", "OR", "AI", "ML", "DL", "NN", "SGD", "FIFO", "SPT", "MWKR", "LoRA"}
SUMMARY_TAIL = r"^(this|the proposed|our)\s+(design|approach|mechanism|strategy|scheme|framework|module)\s+(ensures|guarantees|enables|allows|makes)"
# 30: obscure word -> what the field writes. Any word the venue corpus uses twice is allowed anyway.
OBSCURE = {
    "elucidate": "explain", "delineate": "describe", "epitomize": "typify", "underpin": "support",
    "underpins": "supports", "salient": "main", "myriad": "many", "plethora": "many", "intricate": "complex",
    "nuanced": "(delete or say what differs)", "holistic": "(delete or say what is included)",
    "synergistic": "combined", "synergy": "combined effect", "harness": "use", "harnesses": "uses",
    "bolster": "support", "foster": "encourage", "garner": "gain", "pertinent": "relevant",
    "endeavor": "effort", "ameliorate": "improve", "expedite": "speed up", "necessitate": "require",
    "necessitates": "requires", "encompass": "include", "encompasses": "includes", "albeit": "although",
    "whilst": "while", "hitherto": "so far", "heretofore": "so far", "vis-a-vis": "compared with",
    "vis-\u00e0-vis": "compared with", "in lieu of": "instead of", "paramount": "most important",
    "quintessential": "typical", "multifaceted": "(say which facets)", "meticulously": "carefully",
    "pivotal": "key", "cornerstone": "basis", "landscape": "field", "realm": "field", "tapestry": "(delete)",
    "testament": "evidence", "underscore": "show", "underscores": "shows", "showcase": "show",
    "showcases": "shows", "unravel": "explain", "unveil": "present", "usher": "(delete)",
    "embark": "begin", "pave the way": "enable", "shed light on": "explain", "bridge the gap": "(say what is bridged)",
    "at the forefront": "leading", "cutting-edge": "recent", "groundbreaking": "(delete)",
    "transformative": "(delete)", "commence": "begin", "ascertain": "determine", "exemplify": "show",
    "posit": "propose", "purport": "claim", "substantiate": "support", "corroborate": "confirm",
    "amalgamate": "combine", "juxtapose": "compare", "paradigm shift": "(delete)", "a myriad of": "many",
    "an array of": "several", "hinges on": "depends on", "boils down to": "reduces to", "akin to": "similar to",
    "wherein": "where", "whereby": "by which", "thereof": "(rewrite)", "therein": "(rewrite)",
    "hereinafter": "(rewrite)", "aforementioned": "these", "utmost": "(delete)", "intriguing": "(delete)",
    "elegant": "(delete)", "noteworthy": "(delete)", "burgeoning": "growing", "ubiquitous": "common",
    "seminal": "early", "profound": "(delete)", "immense": "large", "vast": "large", "sheer": "(delete)",
}
ZH_OBSCURE = ("赋能", "抓手", "底层逻辑", "闭环", "颗粒度", "擘画", "彰显", "凸显", "阐释", "诠释", "赋予",
              "不可或缺", "举足轻重", "行之有效", "有力支撑", "深度融合", "全方位", "多维度", "层层递进",
              "应运而生", "蔚然成风", "纵观", "剖析", "探究")
# 31: hyphen chains the field writes as fixed phrases, and prefix words the field writes solid
HYPHEN_OK = {"state-of-the-art", "end-to-end", "out-of-distribution", "out-of-the-box", "off-the-shelf",
             "one-to-one", "one-to-many", "many-to-one", "many-to-many", "peer-to-peer", "first-in-first-out",
             "first-come-first-served", "learning-to-rank", "learning-to-optimize", "in-the-loop",
             "nearest-neighbor-based", "hard-to-solve", "editor-in-chief", "proof-of-concept", "rule-of-thumb",
             "trade-off", "worst-case", "best-of-n"}
PREFIX_SOLID = {"nonlinear", "nonconvex", "nontrivial", "nonnegative", "nonzero", "nonstationary",
                "nondeterministic", "nonuniform", "nonempty", "preprocessing", "preprocess", "subproblem",
                "subproblems", "subset", "subsequence", "subgraph", "suboptimal", "subspace", "subtask", "subtasks",
                "multiobjective", "multiagent", "multimodal", "multistage", "coevolution", "reoptimization",
                "reoptimize", "recompute", "reuse", "semidefinite", "interdependent", "intraday", "postprocessing"}
CAMEL_OK = {"PyTorch", "TensorFlow", "GitHub", "OpenAI", "DeepMind", "LaTeX", "BibTeX", "JavaScript",
            "MATLAB", "LightGBM", "XGBoost", "AdamW", "ResNet", "BatchNorm", "LayerNorm", "OpenReview",
            "ArXiv", "YouTube", "MinMax", "OpenMP", "MuJoCo", "GraphSAGE", "PageRank", "WordNet", "ImageNet"}
# 33: scaffolding openers
SUMMARY_OPENERS = ("in summary", "to summarize", "to sum up", "in conclusion", "taken together",
                   "in this section, we have", "we have shown that", "overall, this section", "all in all")
ROADMAP_OPENERS = ("the remainder of this", "the rest of this", "this section is organized", "this paper is organized",
                   "the remainder of the paper", "the rest of the paper")
ZH_SUMMARY_OPENERS = ("综上", "总之", "总而言之", "总的来说", "本节总结")
ZH_ROADMAP_OPENERS = ("本文其余部分", "本节其余部分", "本文的组织", "本文结构安排", "下文安排")

# ---- word lists: Chinese -------------------------------------------------
ZH_CONNECTIVES = ("此外", "而且", "另外", "值得注意的是", "综上所述", "首先", "其次", "最后", "总之", "与此同时")
ZH_BANNED = ("值得注意的是", "综上所述", "旨在", "本文所提出的")
ZH_HEDGES = ("可能", "或许", "在一定程度上", "某种程度上")
ZH_INTENSIFIERS = ("显著地", "有效地", "极大地", "大幅", "充分地", "全面地")
ZH_GENERIC_VERBS = ("实现了", "有助于", "促进了", "提升了")
ZH_ABSTRACT_NOUNS = ("框架", "范式", "策略", "模块", "体系")
ZH_SUMMARY_TAIL = r"^(这一|该|上述)(设计|机制|做法|策略)(保证|确保|使得)"

# ---- thresholds: built-in defaults, overridden by style_baseline.json ---
TH_DEFAULT = dict(connectives=2, hedges=2, triplets=2, paragraph_leads=4, sent_cv_min=0.35,
                  intensifier_per_k=3.0, generic_verb_per_k=6.0, abstract_noun_per_k=8.0,
                  cluster=4, acronym_min_uses=10, self_acronyms=1,
                  obscure=2, hyphen_per_k=25.0, paren_per_100=1.5, ie_eg=4, enum_groups=1,
                  lists=1, proof_words=120, camel_names=1)


def load_thresholds(baseline, notes: list[str]) -> dict:
    """Built-in defaults, loosened/tightened by style_baseline.json (made by style_baseline.py)."""
    TH = dict(TH_DEFAULT)
    if baseline is not None and Path(baseline).exists():
        base = json.loads(Path(baseline).read_text(encoding="utf-8"))
        for k in ("intensifier_per_k", "generic_verb_per_k", "abstract_noun_per_k", "connectives"):
            if k in base:
                TH[k] = round(base[k] * 1.2, 2)
        if "sent_cv" in base:
            TH["sent_cv_min"] = round(base["sent_cv"] * 0.8, 2)
        TH["_vocab"] = set(base.get("vocabulary", []))          # words the venue itself uses: item 30 lets them pass
        TH["_hyphenated"] = set(base.get("hyphenated", []))     # hyphen spellings the venue uses: item 31 lets them pass
        notes.append(f"baseline: {Path(baseline).name} (published-paper thresholds, +/-20%)")
    TH.setdefault("_vocab", set()); TH.setdefault("_hyphenated", set())
    return TH


def plain_to_texish(text: str) -> str:
    """Markdown / plain text -> the subset of LaTeX the checks understand: fenced code dropped
    (newlines kept), headings dropped, **bold** mapped to \\textbf{}, *italic* to \\emph{}."""
    keep_lines = lambda m: "\n" * m.group(0).count("\n")
    t = re.sub(r"`{3}.*?`{3}", keep_lines, text, flags=re.S)        # fenced code (written without literal triple backticks so the file embeds in Markdown)
    t = re.sub(r"^#{1,6} .*$", "", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![*\w])\*(?!\*)([^*\n]+?)\*(?![*\w])", r"\\emph{\1}", t)
    return t


def run_style_checks(sources, style_sources=None, zh: bool = False, baseline=None, root=None,
                     abstract_text=None, method_name: str | None = None):
    """Items 14-27 over `sources` (whole text: acronym budget, abstract) and `style_sources`
    (per-file items; defaults to `sources`). Returns (problems, notes). `root` is where the
    reverse outline is written. `abstract_text` lets check.py pass the abstract it already parsed."""
    ZH = zh
    SOURCES = [Path(s) for s in sources]
    STYLE_SOURCES = [Path(s) for s in style_sources] if style_sources is not None else SOURCES
    ROOT = Path(root) if root is not None else (SOURCES[0].resolve().parent if SOURCES else Path("."))
    problems: list[str] = []
    notes: list[str] = []
    TH = load_thresholds(baseline, notes)
    outline: list[str] = []

    def read_src(p: Path) -> str:
        t = p.read_text(encoding="utf-8")
        return plain_to_texish(t) if p.suffix.lower() in {".md", ".txt"} else t


    def sentences_of(text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？])" if ZH else r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if len(re.findall(r"\w", s)) >= (6 if ZH else 3)]


    def length_of(sentence: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]", sentence)) if ZH else len(sentence.split())


    def per_k(count: int, total: int) -> float:
        return 1000.0 * count / max(total, 1)


    for src in STYLE_SOURCES:
        raw = read_src(src)
        raw_lines = raw.splitlines()
        prose = prose_of(raw)
        name = src.name
        low = prose.lower()
        sentences = sentences_of(prose)
        n_units = sum(length_of(s) for s in sentences)          # words (en) or characters (zh)

        # 14: prose semicolons (a "% style-exempt" comment on the line excuses it)
        semi = "；" if ZH else ";"
        for i, line in enumerate(prose.splitlines(), 1):
            if semi in line:
                orig = raw_lines[i - 1] if i <= len(raw_lines) else ""
                if "style-exempt" not in orig:
                    problems.append(f"prose semicolon at {name}:{i}")

        # 15: sentence-initial connectives
        conns = ZH_CONNECTIVES if ZH else CONNECTIVES
        hits = [s[:40] for s in sentences if s.lower().startswith(conns)]
        if len(hits) > TH["connectives"]:
            problems.append(f"{name}: {len(hits)} sentence-initial connectives "
                            f"(budget {TH['connectives']}): {hits[:3]}")

        # 16: banned stock phrases
        for phrase in (ZH_BANNED if ZH else BANNED):
            if phrase in low:
                problems.append(f"{name}: banned phrase '{phrase}'")

        # 17: hedging -- budget per section, and stacked hedges are out regardless
        hedges = ZH_HEDGES if ZH else HEDGES
        n_hedges = sum(low.count(h) for h in hedges)
        if n_hedges > TH["hedges"]:
            problems.append(f"{name}: {n_hedges} hedges (budget {TH['hedges']}) -- "
                            f"narrow the claim and cite the number instead of softening it")
        for s_ in sentences:
            sl = s_.lower()
            if ZH:
                if len(re.findall(r"可能|或许|在一定程度上", sl)) >= 2:
                    problems.append(f"{name}: stacked hedge in: {s_[:40]}")
            elif re.search(r"\b(may|might|could)\b", sl) and \
                 re.search(r"\b(potentially|possibly|perhaps|to some extent)\b", sl):
                problems.append(f"{name}: stacked hedge in: {s_[:60]}")

        # 19: colons inside a sentence (a colon that ends the line introduces a
        # display block, list or algorithm and is fine)
        colon = "：" if ZH else ":"
        for i, line in enumerate(prose.splitlines(), 1):
            if colon in line and "style-exempt" not in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
                stripped = line.rstrip().rstrip("\\").rstrip()
                if stripped.endswith(colon):
                    notes.append(f"{name}:{i} line-final colon (introducing a block) -- allowed")
                elif not re.search(r"https?:|\d:\d", line):
                    problems.append(f"mid-sentence colon at {name}:{i}")

        # 20: dashes used as parentheticals (number ranges such as 20--50 are fine)
        for i, line in enumerate(prose.splitlines(), 1):
            if "style-exempt" in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
                continue
            if ZH and ("——" in line or "—" in line):
                problems.append(f"dash in prose at {name}:{i}")
            if not ZH and (re.search(r"---|—", line) or re.search(r"\s--\s", line)):
                problems.append(f"dash in prose at {name}:{i}")

        # 21: emphasis inside sentences; run-in headings (\paragraph) within budget
        bold_at = [i for i, line in enumerate(prose.splitlines(), 1) if "\\textbf{" in line]
        for i in bold_at:
            if "style-exempt" not in (raw_lines[i - 1] if i <= len(raw_lines) else ""):
                problems.append(f"bold inside running text at {name}:{i} "
                                f"(use \\paragraph{{}} for a run-in heading, no bold for emphasis)")
        n_emph = len(re.findall(r"\\(?:emph|textit)\{", prose))
        if n_emph > 1:
            problems.append(f"{name}: {n_emph} italic emphases (budget 1: the first definition of a term)")
        for blk in re.split(r"\\section\*?\{", strip_comments(raw)):
            n_leads = len(re.findall(r"\\paragraph\{", blk))
            if n_leads > TH["paragraph_leads"]:
                title = blk.split("}", 1)[0][:20]
                problems.append(f"{name}: {n_leads} run-in headings in section '{title}' (budget {TH['paragraph_leads']})")
        if ZH:
            for i, line in enumerate(strip_comments(raw).splitlines(), 1):
                if re.match(r"\s*\\textbf\{[^}]{1,12}。\}", line):
                    problems.append(f"bold run-in heading at {name}:{i} -- use \\paragraph{{}} instead")

        # 23: sentence rhythm and triplet lists
        lengths = [length_of(s) for s in sentences]
        if len(lengths) >= 8:
            cv = statistics.pstdev(lengths) / max(statistics.mean(lengths), 1)
            if cv < TH["sent_cv_min"]:
                problems.append(f"{name}: sentence lengths too uniform (CV {cv:.2f} < {TH['sent_cv_min']}) "
                                f"-- vary long and short sentences")
            else:
                notes.append(f"{name}: sentence-length CV {cv:.2f}, mean {statistics.mean(lengths):.1f}")
        if not ZH:
            triplets = re.findall(r"\b\w+, \w+,? and \w+\b", prose)
            if len(triplets) > TH["triplets"]:
                problems.append(f"{name}: {len(triplets)} three-item lists (budget {TH['triplets']}): "
                                f"{triplets[:2]}")

        # 24: lexical density (per 1000 words / characters)
        if ZH:
            inten = sum(prose.count(w) for w in ZH_INTENSIFIERS)
            gener = sum(prose.count(w) for w in ZH_GENERIC_VERBS)
            abstr = sum(prose.count(w) for w in ZH_ABSTRACT_NOUNS)
            if prose.count("本文所提出的") > 2:
                problems.append(f"{name}: '本文所提出的' repeated {prose.count('本文所提出的')} times")
        else:
            inten = sum(len(re.findall(r"\b%s\b" % w, low)) for w in INTENSIFIERS)
            gener = sum(len(re.findall(r"\b%s[sd]?\b" % w, low)) for w in GENERIC_VERBS)
            abstr = sum(len(re.findall(r"\b%ss?\b" % w, low)) for w in ABSTRACT_NOUNS)
            extra = len(re.findall(r"not only .{0,60} but also|\brespectively\b", low))
            if extra > 2:
                notes.append(f"{name}: {extra} 'not only/but also' or 'respectively' (limit 2)")
        for label, cnt, key in (("intensifiers", inten, "intensifier_per_k"),
                                ("generic verbs", gener, "generic_verb_per_k"),
                                ("abstract nouns", abstr, "abstract_noun_per_k")):
            rate = per_k(cnt, n_units)
            if rate > TH[key] and cnt >= 3:
                problems.append(f"{name}: {label} at {rate:.1f}/1000 (limit {TH[key]}) -- "
                                f"replace with the specific action or the number")

        # 25: paragraph specificity (method / experiment sections) and summary tails
        lname = name.lower()
        stripped = strip_comments(raw)
        if "\\begin{document}" in stripped:
            stripped = stripped[stripped.index("\\begin{document}"):]
        # which paragraphs belong to a core (method / experiment / problem) section
        core_titles = re.compile(r"方法|实验|问题|模型|method|experiment|result|problem", re.I)
        chunks, section_core = re.split(r"(\\section\*?\{[^}]*\})", stripped), []
        current = not ZH and any(k in lname for k in ("method", "experiment", "result", "problem"))
        for c in chunks:
            if c.startswith("\\section"):
                current = bool(core_titles.search(c)); continue
            section_core.append((c, current))
        for chunk, is_core in section_core:
            for p in re.split(r"\n\s*\n", chunk):
                ptxt = prose_of(p)
                if len(re.findall(r"[\u4e00-\u9fff]" if ZH else r"[A-Za-z]{3,}", ptxt)) < (60 if ZH else 30):
                    continue
                if "〔填写" in p or "\\draftnote" in p:
                    continue
                first = sentences_of(ptxt)[:1]
                if first:
                    outline.append(f"- ({name}) {first[0][:120]}")
                if not is_core:
                    continue
                concrete = bool(re.search(r"\d", p) or re.search(r"\\(?:eq)?ref\{|\\cite|\$[^$]+\$", p)
                                or re.search(r"例如|譬如|比如|for example|e\.g\.", p))
                if not concrete:
                    problems.append(f"{name}: paragraph with nothing concrete (no number, equation, "
                                    f"table/figure reference, citation or example): '{ptxt.strip()[:50]}'")
                last = sentences_of(ptxt)[-1:] if sentences_of(ptxt) else []
                if last and re.match(ZH_SUMMARY_TAIL if ZH else SUMMARY_TAIL, last[0].strip(), re.I):
                    problems.append(f"{name}: paragraph closes on a summary sentence: '{last[0][:60]}'")

        # 26: citation clusters
        for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", stripped):
            keys = [k for k in m.group(1).split(",") if k.strip()]
            if len(keys) > TH["cluster"]:
                problems.append(f"{name}: citation cluster of {len(keys)} keys -- give each work one "
                                f"specific sentence or cite the survey")

    # 18: naming budget -- acronyms and coined compound names (whole manuscript)
    whole = strip_comments("\n".join(read_src(p) for p in SOURCES))
    defs = re.findall(r"[（(]([A-Z][A-Za-z]{1,5})[)）]", whole)
    seen: dict[str, int] = {}
    for a in defs:
        if a.upper() == a or a in COMMON_ACRONYMS:
            seen.setdefault(a, 0)
    for a in seen:
        seen[a] = len(re.findall(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(a), whole)) - 1
    self_made = {a: n for a, n in seen.items() if a not in COMMON_ACRONYMS}
    notes.append("acronyms: " + ", ".join(f"{a}({n} uses)" for a, n in sorted(seen.items())) if seen else "acronyms: none")
    if len(self_made) > TH["self_acronyms"]:
        weak = [a for a, n in self_made.items() if n < TH["acronym_min_uses"]]
        # one self-made acronym (the method's own name) is free; every further one must earn its keep
        if len(self_made) - 1 > len(self_made) - 1 - len(weak) or len(weak) >= len(self_made):
            problems.append(f"self-made acronyms {sorted(self_made)}: beyond the method's own name, an "
                            f"acronym needs >= {TH['acronym_min_uses']} uses (weak: {weak})")
    for a, n in seen.items():
        if n <= 2:
            notes.append(f"acronym {a} defined but used {n} times -- spell it out and drop it")
    if ZH:
        for m in sorted(set(re.findall(r"[\u4e00-\u9fff]{2,6}(?:感知|驱动|增强|自适应)[\u4e00-\u9fff]{0,4}(?:机制|模块|框架|策略|网络)", whole))):
            problems.append(f"coined compound name: {m} -- name the mechanism with existing terms")
    else:
        for m in sorted(set(re.findall(r"(?<![.!?]\s)(?<!\n)\b((?:[A-Z][a-z]+[- ]){2,}[A-Z][a-z]+)\b", whole))):
            if any(w.lower() in NAME_ADJ for w in re.split(r"[- ]", m)):
                problems.append(f"coined name with an effect adjective: '{m}' -- describe the mechanism, not its merit")
            else:
                notes.append(f"title-case compound '{m}' -- use lower case unless it is the method's name")
        for m in sorted(set(re.findall(r"\b\w+-%s-\w+-%s\b|\b\w+-%s\s+\w+-%s\b" % (SUFFIX, SUFFIX, SUFFIX, SUFFIX), whole))):
            problems.append(f"stacked modifier chain: '{m}'")

    # 22: the abstract is one paragraph
    if ZH:
        m = re.search(r"\\section\*\{摘要\}(.*?)(?=\\vspace|\\noindent\\textbf\{关键词|\\section)", whole, re.S)
        abstract_text = m.group(1) if m else None
    elif abstract_text is None:                                 # standalone use: find the abstract ourselves
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", whole, re.S)
        abstract_text = m.group(1) if m else None
    if abstract_text is not None:
        inner = re.sub(r"^\s*\\noindent", "", abstract_text.strip())
        if re.search(r"\n\s*\n", inner.strip()) or "\\par" in inner:
            problems.append("abstract is broken into paragraphs -- a single paragraph, six sentences")

    # 27: reverse outline for the logic pass
    if outline:
        out = (ROOT / ("reverse-outline-zh.md" if ZH else "reverse-outline.md"))
        out.write_text("# Reverse outline (first sentence of every paragraph)\n\n" + "\n".join(outline) + "\n",
                       encoding="utf-8")
        notes.append(f"reverse outline: {out} ({len(outline)} paragraphs) -- read it top to bottom; "
                     f"it must argue on its own")

    # ---- 30-33: vocabulary, symbols & hyphens, proofs, scaffolding --------------
    REL = r"=|\\le\b|\\leq\b|\\ge\b|\\geq\b|\\neq\b|\\equiv\b|\\subseteq\b|\\subset\b|(?<![<>\\])<(?![<])|(?<![<>\\])>(?![>])"
    DISPLAY = r"\\begin\{(?:align|equation|gather|multline|eqnarray|alignat|flalign)\*?\}|\\\["
    camel_seen: dict[str, int] = {}
    for src in STYLE_SOURCES:
        raw = read_src(src)
        stripped = strip_comments(raw)
        prose = prose_of(raw)
        low = prose.lower()
        name = src.name
        sentences = sentences_of(prose)
        n_words = len(re.findall(r"[A-Za-z]{2,}", prose)) if not ZH else len(re.findall(r"[\u4e00-\u9fff]", prose))

        # 30: obscure vocabulary ------------------------------------------------
        hits: list[str] = []
        if ZH:
            for w in ZH_OBSCURE:
                c = prose.count(w)
                if c:
                    hits += [w] * c
        else:
            for w in OBSCURE:
                if w in TH["_vocab"]:
                    continue                                        # the venue writes it: allowed
                c = len(re.findall(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(w), low))
                if c:
                    hits += [f"{w} -> {OBSCURE[w]}"] * c
        if len(hits) > TH["obscure"]:
            problems.append(f"{name}: {len(hits)} obscure words where the field writes a plain one: "
                            f"{sorted(set(hits))[:6]}")
        elif hits:
            notes.append(f"{name}: obscure word(s) {sorted(set(hits))} -- prefer the field's usual word")

        # 31: symbols and hyphens -----------------------------------------------
        keep_nl = lambda m: "\n" * m.group(0).count("\n")                          # line numbers must match the source
        body_nd = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % MATH_ENVS, keep_nl, stripped, flags=re.S)   # keep inline math
        body_nd = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]", keep_nl, body_nd, flags=re.S)
        for mac, n in STRUCTURAL_MACROS.items():                                     # captions, figure placeholders, notes are not prose
            body_nd = strip_macro(body_nd, mac, n)
        for i, line in enumerate(body_nd.splitlines(), 1):
            exempt = "style-exempt" in line
            if exempt:
                continue
            if re.search(r"[→⇒↑↓⇐⇔]", line) or re.search(r"\$\s*\\(?:to|rightarrow|Rightarrow|longrightarrow|uparrow|downarrow|nearrow|searrow)\s*\$", line):
                problems.append(f"arrow standing in for words at {name}:{i} -- write 'leads to' / 'higher is better'")
            if re.search(r"[×÷≈≤≥]", line) or re.search(r"(?<![\w$])[+\u2212]\d+(?:\.\d+)?\s*(?:%|pp\b|points?\b)", line) \
                    or re.search(r"\b\d+(?:\.\d+)?\s*[×x]\s*(?:faster|slower|fewer|more|larger|smaller|speed)", line) \
                    or re.search(r"\$\s*\d+(?:\.\d+)?\s*\\times\s*\$", line):
                problems.append(f"operator standing in for words at {name}:{i} -- 'three times faster', 'an increase of 2.1 points'")
            if "±" in line and not re.search(r"\d\s*±\s*\d", line):
                problems.append(f"'±' outside a mean±std pair at {name}:{i}")
            if re.search(r"(?<!\d)~\s*\d", line):
                notes.append(f"'~' for 'about' at {name}:{i} -- write 'about'")
        # tables: direction arrows and +/- markers belong in the caption, not the cells
        for m in re.finditer(r"\\begin\{(tabular\*?|tabularx|longtable)\}(.*?)\\end\{\1\}", stripped, re.S):
            tbl = m.group(2)
            line = stripped[: m.start()].count("\n") + 1
            if re.search(r"[→⇒↑↓]|\\(?:uparrow|downarrow|nearrow|searrow|rightarrow)\b", tbl):
                problems.append(f"{name}:{line} table uses arrow glyphs -- say 'lower is better' in the caption")
            if re.search(r"\(\s*[+\u2212-]\s*\)|\+/[\u2212-]|(?<![\w$])[+\u2212]\d", tbl):
                notes.append(f"{name}:{line} table carries +/- markers -- give the quantity its own column with a unit")
        if not ZH:
            chains = [w for w in re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+){2,}\b", prose) if w.lower() not in HYPHEN_OK]
            for w in sorted(set(chains)):
                problems.append(f"{name}: hyphen chain '{w}' -- rewrite as a phrase or a defined term")
            for m in re.finditer(r"\b(non|pre|re|sub|multi|co|semi|anti|inter|intra|post|over|under)-([a-z]{3,})\b", low):
                hy, solid = m.group(0), m.group(1) + m.group(2)
                if hy in TH["_hyphenated"]:
                    continue
                if solid in PREFIX_SOLID or solid in TH["_vocab"]:
                    notes.append(f"{name}: '{hy}' -- the field writes '{solid}'")
            n_hy = len(re.findall(r"\b[A-Za-z]+-[A-Za-z]+\b", prose))
            if n_words >= 200 and per_k(n_hy, n_words) > TH["hyphen_per_k"]:
                notes.append(f"{name}: {per_k(n_hy, n_words):.0f} hyphenated compounds per 1000 words (limit {TH['hyphen_per_k']})")
            for cm in re.findall(r"(?<![\\\w])[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", prose):      # not \MacroNames
                if cm not in CAMEL_OK and cm != method_name:
                    camel_seen[cm] = camel_seen.get(cm, 0) + 1

        # 32: proofs and derivations ---------------------------------------------
        for m in re.finditer(r"\\begin\{proof\}(.*?)\\end\{proof\}", stripped, re.S):
            body = m.group(1)
            line = stripped[: m.start()].count("\n") + 1
            n_disp = len(re.findall(DISPLAY, body))
            words = len(re.findall(r"[A-Za-z]{2,}", prose_of(body))) if not ZH else len(re.findall(r"[\u4e00-\u9fff]", prose_of(body)))
            if words > TH["proof_words"] and n_disp == 0:
                problems.append(f"{name}:{line} proof of {words} words with no displayed derivation -- "
                                f"set the steps in an align environment, one step per line")
            for im in re.finditer(r"\$([^$]+)\$", body):
                if len(re.findall(REL, im.group(1))) >= 2:
                    problems.append(f"{name}:{line} inline relation chain inside a proof: ${im.group(1)[:50]}$ -- one step per line in align")
                    break
        for im in re.finditer(r"\$([^$]+)\$", body_nd):
            if len(re.findall(REL, im.group(1))) >= 3:
                ln = body_nd[: im.start()].count("\n") + 1
                problems.append(f"{name}:{ln} inline chain of {len(re.findall(REL, im.group(1)))} relations -- a derivation belongs in align, not in a sentence")
        for m in re.finditer(r"\\begin\{(align|alignat|flalign|eqnarray|gather)\*?\}(.*?)\\end\{\1\*?\}", stripped, re.S):
            blk = m.group(2)
            if blk.count("\\\\") >= 2 and not re.search(r"\\label|\\text\{|&&|\\tag", blk):
                notes.append(f"{name}:{stripped[: m.start()].count(chr(10)) + 1} {blk.count(chr(92) * 2) + 1}-line derivation with no step justification or label -- add \\text{{(by ...)}} or a label per step")

        # 33: scaffolding ---------------------------------------------------------
        groups = 0
        if ZH:
            groups += len(re.findall(r"（一）.*?（二）", prose, re.S)) + len(re.findall(r"（1）.*?（2）.*?（3）", prose, re.S))
            groups += len(re.findall(r"首先.{0,200}?其次", prose, re.S))
        else:
            groups += len(re.findall(r"\(i\).*?\(ii\).*?\(iii\)", low, re.S))
            groups += len(re.findall(r"\(1\).{1,300}?\(2\).{1,300}?\(3\)", low, re.S))
            groups += len(re.findall(r"\b(?:first|firstly),.{1,400}?\b(?:second|secondly),", low, re.S))
        if groups > TH["enum_groups"]:
            problems.append(f"{name}: {groups} enumerated sequences ((i)(ii)(iii) / First, Second, ...) -- "
                            f"budget {TH['enum_groups']}; let the order of the sentences carry the structure")
        sec_title = ""
        for chunk in re.split(r"(\\section\*?\{[^}]*\})", stripped):
            if chunk.startswith("\\section"):
                sec_title = chunk.lower(); continue
            in_intro = "intro" in sec_title or "intro" in name.lower() or "引言" in sec_title or "绪论" in sec_title
            in_concl = any(k in sec_title or k in name.lower() for k in ("conclu", "discussion", "结论", "总结", "讨论"))
            for para in re.split(r"\n\s*\n", chunk):
                ptxt = prose_of(para).strip()
                if not ptxt:
                    continue
                head = ptxt[:60].lower()
                if (head.startswith(ZH_SUMMARY_OPENERS if ZH else SUMMARY_OPENERS)) and not in_concl:
                    problems.append(f"{name}: summary paragraph outside the conclusion: '{ptxt[:50]}'")
                if (head.startswith(ZH_ROADMAP_OPENERS if ZH else ROADMAP_OPENERS)) and not in_intro:
                    problems.append(f"{name}: roadmap paragraph outside the introduction: '{ptxt[:50]}'")
        if not ZH and n_words >= 100:
            parens = len(re.findall(r"\((?!\s*\d{4}\))", prose))
            if 100.0 * parens / n_words > TH["paren_per_100"]:
                notes.append(f"{name}: {parens} parentheticals in {n_words} words -- cut asides or make them sentences")
            ie = len(re.findall(r"\b(?:i\.e\.|e\.g\.|etc\.)", low))
            if ie > TH["ie_eg"]:
                notes.append(f"{name}: {ie} i.e./e.g./etc. (limit {TH['ie_eg']}) -- say 'for example' or name the thing")
        n_lists = len(re.findall(r"\\begin\{(?:itemize|enumerate)\}", stripped))
        if n_lists > TH["lists"]:
            notes.append(f"{name}: {n_lists} bullet lists in running text -- prose sections carry structure in sentences")
    if len(camel_seen) > TH["camel_names"]:
        problems.append(f"CamelCase coined names {sorted(camel_seen)} -- only the method's own name may be one "
                        f"(pass --method-name); the rest are ordinary terms in lower case")
    elif camel_seen:
        notes.append(f"CamelCase name {sorted(camel_seen)} taken as the method's name; pass --method-name to confirm")

    return problems, notes


def looks_chinese(paths) -> bool:
    text = "".join(Path(p).read_text(encoding="utf-8", errors="replace") for p in paths)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk > 0.3 * max(len(re.findall(r"[\u4e00-\u9fffA-Za-z]", text)), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help=".tex / .md / .txt files; sections are checked one by one")
    ap.add_argument("--zh", action="store_true", help="Chinese text (default: auto-detect)")
    ap.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parent / "style_baseline.json")
    ap.add_argument("--outline-dir", type=Path, default=None, help="where reverse-outline.md goes (default: first file's folder)")
    ap.add_argument("--method-name", default=None, help="the one CamelCase name allowed (the method's own name)")
    a = ap.parse_args()
    zh = a.zh or looks_chinese(a.files)
    problems, notes = run_style_checks(a.files, zh=zh, baseline=a.baseline,
                                       root=a.outline_dir or a.files[0].resolve().parent, method_name=a.method_name)
    print(f"files   : {len(a.files)}" + (" (zh)" if zh else ""))
    for n in notes:
        print(f"  note   {n}")
    for p in problems:
        print(f"  FAIL   {p}")
    print()
    if problems:
        print(f"{len(problems)} problem(s) found.")
        sys.exit(1)
    print("all style checks passed.")

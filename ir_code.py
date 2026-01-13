import re, math, xml.etree.ElementTree as ET, time, argparse, os, sys, urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union, IO
from nltk.stem import PorterStemmer

TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _read_all(src: Union[str, IO[str]], encoding: str = "utf-8", errors: str = "ignore") -> str:
    if hasattr(src, "read"):
        return src.read()
    s = str(src)
    if s == "-":
        return sys.stdin.read()
    if _is_url(s):
        with urllib.request.urlopen(s) as r:
            data = r.read()
        try:
            return data.decode(encoding, errors=errors)
        except Exception:
            return data.decode("utf-8", errors=errors)
    with open(s, "r", encoding=encoding, errors=errors) as f:
        return f.read()

def _iter_lines(src: Union[str, IO[str]], encoding: str = "utf-8", errors: str = "ignore"):
    if hasattr(src, "read"):
        for line in src:
            yield line
        return
    s = str(src)
    if s == "-":
        for line in sys.stdin:
            yield line
        return
    if _is_url(s):
        with urllib.request.urlopen(s) as r:
            for raw in r:
                try:
                    yield raw.decode(encoding, errors=errors)
                except Exception:
                    yield raw.decode("utf-8", errors=errors)
        return
    with open(s, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            yield line

def _write_lines(dst: Union[str, IO[str]], lines):
    if hasattr(dst, "write"):
        for x in lines:
            dst.write(x)
        return
    s = str(dst)
    if s == "-":
        for x in lines:
            sys.stdout.write(x)
        return
    with open(s, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x)

def load_stopwords(src: Union[str, IO[str]]) -> Set[str]:
    s = set()
    text = _read_all(src, encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        w = line.strip().lower()
        if w:
            s.add(w)
    return s

class Preprocessor:
    def __init__(self, stopwords_src: Union[str, IO[str], None]):
        self.stops = load_stopwords(stopwords_src) if stopwords_src is not None else set()
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str) -> List[str]:
        return [m.group(0) for m in TOKEN_PATTERN.finditer(text.lower())]

    def _norm(self, tok: str) -> str:
        return tok.replace("’", "'").replace("'", "").replace("-", "")

    def preprocess_doc(self, text: str) -> List[Tuple[str, int]]:
        out, pos = [], 0
        for tok in self.tokenize(text):
            if tok in self.stops:
                continue
            base = self._norm(tok)
            if not base:
                continue
            out.append((self.stemmer.stem(base), pos))
            pos += 1
        return out

    def preprocess_query_terms(self, text: str) -> List[str]:
        out = []
        for tok in self.tokenize(text):
            if tok in self.stops:
                continue
            base = self._norm(tok)
            if not base:
                continue
            out.append(self.stemmer.stem(base))
        return out

def parse_qid_and_text(line: str) -> Optional[Tuple[str, str]]:
    if line is None:
        return None
    s = line.strip().lstrip("\ufeff")
    if not s or s.startswith("#"):
        return None
    m = re.match(r"\s*(\d+)\s+(.*\S)\s*$", s)
    if m:
        return m.group(1), m.group(2)
    parts = s.split(None, 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[0], parts[1]
    return None

def _wrap_root(s: str) -> str:
    return "<ROOT>\n" + s + "\n</ROOT>"

def read_and_preprocess(xml_src: Union[str, IO[str]], pre: Preprocessor) -> Dict[str, List[Tuple[str, int]]]:
    raw = _read_all(xml_src, encoding="utf-8", errors="ignore")
    root = ET.fromstring(_wrap_root(raw))
    data: Dict[str, List[Tuple[str, int]]] = {}
    for doc in root.findall(".//DOC"):
        docno = (doc.findtext("DOCNO") or "").strip()
        h, t = doc.find("HEADLINE"), doc.find("TEXT")
        head = " ".join(x.strip() for x in h.itertext()) if h is not None else ""
        body = " ".join(x.strip() for x in t.itertext()) if t is not None else ""
        full = (head + " " + body).strip()
        data[docno] = pre.preprocess_doc(full)
    return data

def save_pre_result(processed: Dict[str, List[Tuple[str, int]]], out_dst: Union[str, IO[str]]):
    def gen():
        for docno, pairs in processed.items():
            payload = " ".join(f"{t}:{p}" for t, p in pairs)
            yield f"{docno}\t{payload}\n"
    _write_lines(out_dst, gen())

class PositionalInvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.df: Dict[str, int] = {}
        self.N = 0
        self.doc_len: Dict[str, int] = {}

    def add_document(self, docno: str, items: List[Tuple[str, int]]):
        self.doc_len[docno] = len(items)
        for term, pos in items:
            self.index[term][docno].append(pos)

    def finalize(self):
        for term, pl in self.index.items():
            for d in pl:
                pl[d].sort()
        self.df = {t: len(pl) for t, pl in self.index.items()}
        self.N = len(self.doc_len)

    def save(self, out_dst: Union[str, IO[str]]):
        def gen():
            for term in sorted(self.index.keys()):
                yield f"{term}:{len(self.index[term])}\n"
                for docno in sorted(self.index[term].keys()):
                    pos_str = ", ".join(str(p) for p in self.index[term][docno])
                    yield f"\t{docno}: {pos_str}\n"
        _write_lines(out_dst, gen())

def build_index_from_pre(pre_src: Union[str, IO[str]], out_dst: Union[str, IO[str], None] = None) -> PositionalInvertedIndex:
    docs: Dict[str, List[Tuple[str, int]]] = {}
    for raw in _iter_lines(pre_src, encoding="utf-8", errors="ignore"):
        line = raw.strip()
        if not line:
            continue
        docno, rest = line.split("\t", 1) if "\t" in line else (line, "")
        pairs: List[Tuple[str, int]] = []
        for chunk in rest.split():
            if ":" not in chunk:
                continue
            t, p = chunk.split(":", 1)
            try:
                pairs.append((t, int(p)))
            except ValueError:
                pass
        docs[docno] = pairs
    idx = PositionalInvertedIndex()
    for d, pairs in docs.items():
        idx.add_document(d, pairs)
    idx.finalize()
    if out_dst is not None:
        idx.save(out_dst)
    return idx

def load_index(index_src: Union[str, IO[str]]):
    postings: Dict[str, Dict[str, List[int]]] = {}
    all_docs: Set[str] = set()
    current = None
    for raw in _iter_lines(index_src, encoding="utf-8", errors="ignore"):
        line = raw.rstrip("\n")
        if not line:
            continue
        if not line.startswith("\t"):
            current = line.split(":", 1)[0].strip()
            postings[current] = {}
            continue
        body = line.strip()
        doc, pos_part = body.split(":", 1)
        doc = doc.strip()
        pos_list: List[int] = []
        for x in pos_part.split(","):
            x = x.strip()
            if x:
                try:
                    pos_list.append(int(x))
                except ValueError:
                    pass
        postings[current][doc] = pos_list
        all_docs.add(doc)
    return postings, all_docs

class BoolEngine:
    def __init__(self, postings, all_docs, pre: Preprocessor):
        self.postings = postings
        self.all_docs = all_docs
        self.pre = pre

    def _docs(self, term: str) -> Set[str]:
        return set(self.postings.get(term, {}).keys())

    def _phrase(self, terms: List[str]) -> Set[str]:
        if not terms:
            return set()
        if len(terms) == 1:
            return self._docs(terms[0])
        dicts = [self.postings.get(t, {}) for t in terms]
        cand = set(dicts[0].keys())
        for d in dicts[1:]:
            cand &= set(d.keys())
        hits = set()
        for doc in cand:
            sets = [set(d[doc]) for d in dicts]
            ok = False
            for p0 in dicts[0][doc]:
                prev = p0
                good = True
                for i in range(1, len(sets)):
                    if (prev + 1) not in sets[i]:
                        good = False
                        break
                    prev += 1
                if good:
                    ok = True
                    break
            if ok:
                hits.add(doc)
        return hits

    def _prox(self, a: str, b: str, k: int) -> Set[str]:
        p1 = self.postings.get(a, {})
        p2 = self.postings.get(b, {})
        cand = set(p1.keys()) & set(p2.keys())
        res = set()
        for d in cand:
            i = j = 0
            L1, L2 = p1[d], p2[d]
            while i < len(L1) and j < len(L2):
                if abs(L1[i] - L2[j]) <= k:
                    res.add(d)
                    break
                if L1[i] < L2[j]:
                    i += 1
                else:
                    j += 1
        return res

    def evaluate(self, raw: str) -> List[str]:
        phrases: List[str] = []
        proxes: List[Tuple[int, str, str]] = []

        def _ph_repl(m):
            phrases.append(m.group(1))
            return f"PH_{len(phrases)-1}"

        def _px_repl(m):
            k = int(m.group(1))
            a = m.group(2)
            b = m.group(3)
            proxes.append((k, a, b))
            return f"PX_{len(proxes)-1}"

        s = re.sub(r'"([^"]+)"', _ph_repl, raw)
        s = re.sub(r'#\s*(\d+)\s*\(\s*([^) ,]+)\s*,\s*([^) ,]+)\s*\)?', _px_repl, s, flags=re.I)

        if "#" in raw and "PX_" not in s:
            m = re.search(r'#\s*(\d+)\s*\(\s*([^) ,]+)\s*,\s*([^) ,]+)\s*\)?', raw, flags=re.I)
            if m:
                k = int(m.group(1))
                a = m.group(2)
                b = m.group(3)
                proxes.append((k, a, b))
                s = re.sub(r'#\s*(\d+)\s*\(\s*([^) ,]+)\s*,\s*([^) ,]+)\s*\)?', f"PX_{len(proxes)-1}", s, flags=re.I)
            else:
                return []

        raw_tokens = re.findall(r"\w+|[()]", s)
        tokens: List = []
        for tk in raw_tokens:
            up = tk.upper()
            if up in ("AND", "OR", "NOT", "(", ")"):
                tokens.append(up)
            elif tk.startswith("PH_"):
                idx = int(tk.split("_")[1])
                terms = self.pre.preprocess_query_terms(phrases[idx])
                tokens.append(("PHRASE", terms))
            elif tk.startswith("PX_"):
                idx = int(tk.split("_")[1])
                k, a, b = proxes[idx]
                ta = self.pre.preprocess_query_terms(a)
                tb = self.pre.preprocess_query_terms(b)
                if ta and tb:
                    tokens.append(("PROX", k, ta[0], tb[0]))
                else:
                    tokens.append(set())
            else:
                for t in self.pre.preprocess_query_terms(tk):
                    tokens.append(t)

        def prec(op): return 3 if op == "NOT" else 2 if op == "AND" else 1 if op == "OR" else 0

        out, st = [], []
        for tok in tokens:
            if tok == "(":
                st.append(tok)
            elif tok == ")":
                while st and st[-1] != "(":
                    out.append(st.pop())
                if st and st[-1] == "(":
                    st.pop()
            elif tok in ("AND", "OR", "NOT"):
                while st and st[-1] != "(" and prec(st[-1]) >= prec(tok):
                    out.append(st.pop())
                st.append(tok)
            else:
                out.append(tok)
        while st:
            out.append(st.pop())

        U = self.all_docs
        stack: List[Set[str]] = []
        for tok in out:
            if tok in ("AND", "OR", "NOT"):
                if tok == "NOT":
                    a = stack.pop() if stack else set()
                    stack.append(U - a)
                else:
                    b = stack.pop() if stack else set()
                    a = stack.pop() if stack else set()
                    stack.append(a & b if tok == "AND" else a | b)
            elif isinstance(tok, tuple) and tok[0] == "PHRASE":
                stack.append(self._phrase(tok[1]))
            elif isinstance(tok, tuple) and tok[0] == "PROX":
                _, k, a, b = tok
                stack.append(self._prox(a, b, k))
            else:
                stack.append(self._docs(tok if isinstance(tok, str) else ""))

        res = list(stack.pop() if stack else [])
        res.sort()
        return res

def prepare_tfidf(postings: Dict[str, Dict[str, List[int]]], all_docs: Set[str]):
    N = len(all_docs) or 1
    idf = {t: (math.log10(N / len(pl)) if len(pl) > 0 else 0.0) for t, pl in postings.items()}
    return idf, None

def ranked_search(q, postings, idf, _, pre: Preprocessor, topk=150):
    terms = set(pre.preprocess_query_terms(q))
    if not terms:
        return []
    scores = defaultdict(float)
    for t in terms:
        idf_t = idf.get(t, 0.0)
        if idf_t <= 0:
            continue
        for d, pos in postings.get(t, {}).items():
            tf = len(pos)
            scores[d] += (1 + math.log10(tf)) * idf_t
    res = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return res[:topk]

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--xml", default=os.environ.get("IR_XML", "-"))
    ap.add_argument("--stopwords", default=os.environ.get("IR_STOPWORDS", None))
    ap.add_argument("--pre-out", default=os.environ.get("IR_PRE_OUT", None))
    ap.add_argument("--index-out", default=os.environ.get("IR_INDEX_OUT", None))
    ap.add_argument("--index-in", default=os.environ.get("IR_INDEX_IN", None))
    ap.add_argument("--bool-queries", default=os.environ.get("IR_BOOL_QUERIES", None))
    ap.add_argument("--bool-out", default=os.environ.get("IR_BOOL_OUT", "-"))
    ap.add_argument("--rank-queries", default=os.environ.get("IR_RANK_QUERIES", None))
    ap.add_argument("--rank-out", default=os.environ.get("IR_RANK_OUT", "-"))
    ap.add_argument("--topk", type=int, default=int(os.environ.get("IR_TOPK", "150")))
    args = ap.parse_args()

    t0 = time.time()
    pre = Preprocessor(args.stopwords)

    if args.index_in is not None:
        postings, all_docs = load_index(args.index_in)
    else:
        processed = read_and_preprocess(args.xml, pre)
        if args.pre_out is not None:
            save_pre_result(processed, args.pre_out)
            idx = build_index_from_pre(args.pre_out, args.index_out if args.index_out is not None else None)
        else:
            idx = PositionalInvertedIndex()
            for d, pairs in processed.items():
                idx.add_document(d, pairs)
            idx.finalize()
            if args.index_out is not None:
                idx.save(args.index_out)
        if args.index_out is not None:
            postings, all_docs = load_index(args.index_out)
        else:
            postings = idx.index
            all_docs = set(idx.doc_len.keys())

    if args.bool_queries is not None:
        be = BoolEngine(postings, all_docs, pre)
        out_lines = []
        for raw in _iter_lines(args.bool_queries, encoding="utf-8", errors="ignore"):
            parsed = parse_qid_and_text(raw)
            if parsed is None:
                continue
            qid, qtext = parsed
            docs = be.evaluate(qtext)
            for d in docs:
                out_lines.append(f"{qid}, {d}\n")
        _write_lines(args.bool_out, out_lines)

    if args.rank_queries is not None:
        idf, docnorm = prepare_tfidf(postings, all_docs)
        out_lines = []
        for raw in _iter_lines(args.rank_queries, encoding="utf-8", errors="ignore"):
            parsed = parse_qid_and_text(raw)
            if parsed is None:
                continue
            qid, qtext = parsed
            ranked = ranked_search(qtext, postings, idf, docnorm, pre, topk=args.topk)
            for d, s in ranked:
                out_lines.append(f"{qid},{d},{s:.4f}\n")
        _write_lines(args.rank_out, out_lines)

    _ = time.time() - t0

if __name__ == "__main__":
    main()

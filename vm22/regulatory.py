"""Search over the NAIC Valuation Manual (or any regulatory PDF the user loads).

Pages are split into overlapping chunks, tagged with the VM chapter that the
page belongs to (e.g. "VM-22"), and ranked with TF-IDF cosine similarity.
"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

CHAPTER_RE = re.compile(r"\bVM-(\d{2})\b")


def _chapter(text):
    counts = Counter(CHAPTER_RE.findall(text))
    return f"VM-{counts.most_common(1)[0][0]}" if counts else "other"


def _chunks(text, size=1200, overlap=200):
    text = re.sub(r"[ \t]+", " ", text).strip()
    step = size - overlap
    return [text[i:i + size] for i in range(0, max(len(text) - overlap, 1), step)]


class RegulatoryIndex:
    def __init__(self, records):
        self.records = records
        self.vectorizer = None
        self.matrix = None
        if records:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            self.matrix = self.vectorizer.fit_transform([r["text"] for r in records])

    @classmethod
    def from_pages(cls, pages, document):
        """pages: list of page texts (index 0 = page 1)."""
        records = []
        for i, page in enumerate(pages, start=1):
            chap = _chapter(page)
            for chunk in _chunks(page):
                if len(chunk) > 80:
                    records.append({"document": document, "page": i, "chapter": chap, "text": chunk})
        return cls(records)

    @classmethod
    def from_folder(cls, folder):
        records = []
        for pdf in sorted(Path(folder).glob("*.pdf")):
            pages = [p.extract_text() or "" for p in PdfReader(pdf).pages]
            records += cls.from_pages(pages, pdf.name).records
        return cls(records)

    @property
    def documents(self):
        c = Counter((r["document"], r["chapter"]) for r in self.records)
        docs = {}
        for (doc, chap), n in c.items():
            docs.setdefault(doc, {})[chap] = n
        return docs

    def search(self, query, chapter="any", top_k=5):
        if not self.records:
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if chapter and chapter != "any":
            scores = np.where([r["chapter"] == chapter for r in self.records], scores, -1)
        order = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[i], "score": round(float(scores[i]), 3)} for i in order if scores[i] > 0]

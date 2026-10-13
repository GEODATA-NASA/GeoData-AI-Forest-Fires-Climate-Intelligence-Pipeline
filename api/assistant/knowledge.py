"""Project knowledge base for retrieval: README sections plus a data dictionary.

Retrieval is TF-IDF over heading-sized chunks (scikit-learn is already a
dependency, so no embedding service is needed). The corpus is small and
changes only when the README does, so it is built once per process.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DICTIONARY = """
# Data dictionary: fire detections (NASA FIRMS)
Each row of fire_events is one satellite hot-spot detection, not one fire. A large fire produces many detections per overpass.
- latitude, longitude: centre of the detection pixel, decimal degrees.
- acq_date, acq_time: acquisition date and time in UTC (acq_time is HHMM).
- brightness: brightness temperature of the fire pixel in Kelvin (MODIS channel 21/22, VIIRS I-4).
- bright_t31: background brightness temperature in Kelvin (MODIS channel 31, VIIRS I-5).
- frp: fire radiative power in megawatts (MW), the energy released by the fire within the pixel. It is the basis of the severity label.
- scan, track: pixel size in km along scan and along track. MODIS pixels are about 1 km at nadir and grow toward the swath edge; VIIRS pixels are about 375 m. Larger pixels integrate more fire, so FRP per detection is higher for MODIS.
- confidence: detection confidence 0-100. VIIRS reports low/nominal/high, stored here as 25/50/100.
- daynight: D for day, N for night overpass.
- type: inferred hot-spot type: 0 presumed vegetation fire, 1 active volcano, 2 other static land source, 3 offshore.
- satellite_source: modis (Terra and Aqua), viirs_j1 (NOAA-20), viirs_suomi (Suomi NPP).

# Data dictionary: weather (NASA POWER)
Daily values on the POWER grid (0.5 x 0.625 degrees, thinned to about 1 x 1.25 degrees near fires).
- RH2M: relative humidity at 2 m, percent.
- PRECTOTCORR: bias-corrected total precipitation, mm/day.
- GWETTOP: surface soil wetness, 0 (dry) to 1 (saturated), top 5 cm.
A fire is matched to the nearest same-day weather point within MAX_WEATHER_DISTANCE_KM (default 100 km).

# Severity definition
A detection is "severe" (high FRP) when frp >= HIGH_FRP_THRESHOLD, 10 MW by default. The model predicts the probability
that an already detected fire is severe. It does not predict whether a fire will start. The decision threshold
(default 0.5) turns the probability into a label.
"""


@dataclass
class Chunk:
    title: str
    text: str
    source: str


def _split_markdown(text: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title, lines = "Introduction", []
    for line in text.splitlines():
        heading = re.match(r"^#{1,3}\s+(.*)", line)
        if heading:
            if "".join(lines).strip():
                chunks.append(Chunk(title, "\n".join(lines).strip(), source))
            title, lines = heading.group(1).strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        chunks.append(Chunk(title, "\n".join(lines).strip(), source))
    # Very long sections are split so a hit returns a focused passage.
    focused: list[Chunk] = []
    for chunk in chunks:
        paragraphs = re.split(r"\n\s*\n", chunk.text)
        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) > 1800 and buffer:
                focused.append(Chunk(chunk.title, buffer.strip(), source))
                buffer = ""
            buffer += paragraph + "\n\n"
        if buffer.strip():
            focused.append(Chunk(chunk.title, buffer.strip(), source))
    return focused


class KnowledgeBase:
    def __init__(self, documents: dict[str, str]):
        self.chunks: list[Chunk] = []
        for source, text in documents.items():
            self.chunks.extend(_split_markdown(text, source))
        corpus = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks] or [""]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(corpus)

    @classmethod
    def from_project(cls, root: Path) -> "KnowledgeBase":
        documents = {"data dictionary": DATA_DICTIONARY}
        readme = Path(root) / "README.md"
        if readme.is_file():
            documents["README.md"] = readme.read_text(encoding="utf-8")
        return cls(documents)

    def search(self, query: str, limit: int = 4) -> list[dict]:
        if not query.strip() or not self.chunks:
            return []
        scores = (self._matrix @ self._vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores)[:limit]
        return [
            {"source": self.chunks[i].source, "section": self.chunks[i].title, "score": round(float(scores[i]), 3), "text": self.chunks[i].text}
            for i in order
            if scores[i] > 0
        ]

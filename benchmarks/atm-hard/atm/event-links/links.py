"""Link named-trip expense evidence to source-grounded event anchors."""
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
import unicodedata

EVENT = re.compile(r"\b([A-Z][A-Za-z0-9-]{2,})[ -]+(20\d{2})\b")
FARE_QUESTION = re.compile(r"\b(?:taxi|cab|uber)\b", re.I)
COST_QUESTION = re.compile(r"\b(?:cost|fare|spent|pay|paid|reimburs\w*|expense\w*)\b", re.I)
RIDE = re.compile(r"\b(?:taxi\w*|uber|cab|ride receipt)\b", re.I)
RECEIPT = re.compile(r"\b(?:receipt|fare|transaction|payment|importe|factura|recibo)\b", re.I)
ROUTE = re.compile(r"\b(?:pickup|drop.off|origen|destino|destination|start address|end address)\b", re.I)
MONTHS = "January February March April May June July August September October November December".split()
DATE_RANGE = re.compile(r"\b(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(20\d{2})\b", re.I)
TEXT_FIELDS = ("detail", "short_summary", "caption", "short_caption", "ocr_text")
PLACE_WORDS = re.compile(r"\b(?:calle|avenida|rua|rue|street|road|avenue|plaza|piazza|square|largo|san|santa|saint|st)\s+(?:[a-z]{2,}\s+)?[a-z]{3,}\b")


@dataclass(frozen=True)
class Source:
    day: date
    city: str
    location: str
    fields: dict


def normalized(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def load_sources(data):
    sources = {}
    for kind, filename in (("email", "emails"), ("image", "image_batch_results"), ("video", "video_batch_results")):
        for row in json.loads((data / f"{filename}.json").read_bytes()):
            key = row["id"] if kind == "email" else Path(row[f"{kind}_path"]).stem
            try:
                day = date.fromisoformat(row["timestamp"][:10])
            except (KeyError, ValueError, TypeError):
                continue
            fields = {f: row[f] for f in TEXT_FIELDS if isinstance(row.get(f), str) and row[f]}
            sources[key] = Source(day, (row.get("city") or "").split(",")[0].strip(),
                                  row.get("location_name") or "", fields)
    return sources


def match_in_fields(source, pattern):
    for field, text in source.fields.items():
        match = pattern.search(text)
        if match:
            return {"field": field, "quote": match.group(), "start": match.start(), "end": match.end()}
    return None


def find_anchor(question, baseline, pool, sources):
    event_match = EVENT.search(question)
    if not event_match or not FARE_QUESTION.search(question) or not COST_QUESTION.search(question):
        return None
    name, year = event_match.groups()
    marker = re.compile(r"\b" + re.escape(name) + r"\b", re.I)
    for key in dict.fromkeys([*baseline, *pool]):
        source = sources.get(key)
        if not source or not source.city or source.day.year != int(year):
            continue
        name_at = match_in_fields(source, marker)
        date_at = match_in_fields(source, DATE_RANGE)
        if not name_at or not date_at:
            continue
        first, last, month, range_year = DATE_RANGE.fullmatch(date_at["quote"]).groups()
        if range_year != year:
            continue
        try:
            first, last = (date(int(year), MONTHS.index(month.title()) + 1, int(day)) for day in (first, last))
        except ValueError:
            continue
        if first <= source.day <= last and (last - first).days <= 14:
            return {"event": event_match.group(), "source_id": key, "city": source.city,
                    "start_date": first.isoformat(), "end_date": last.isoformat(),
                    "name_span": name_at, "date_span": date_at}
    return None


def phrases(source):
    for field, text in (*source.fields.items(), ("location_name", source.location)):
        for match in PLACE_WORDS.finditer(normalized(text)):
            words = match.group().split()
            # Use the specific first place phrase, independent of the trailing prose.
            if words[1] not in {"de", "da", "do", "del", "of"}:
                yield " ".join(words[:2]), field
            elif len(words) == 3:
                yield " ".join(words), field


def link_event(question, baseline, pool, sources, facts):
    anchor = find_anchor(question, baseline, pool, sources)
    if not anchor:
        return None, []
    start, end = (date.fromisoformat(anchor[field]) for field in ("start_date", "end_date"))
    city = normalized(anchor["city"])
    eligible = []
    for key in pool:
        source = sources.get(key)
        if not source or key not in facts or not start <= source.day <= end:
            continue
        same_city = normalized(source.city) == city if source.city else any(
            re.search(r"\b" + re.escape(city) + r"\b", normalized(text)) for text in source.fields.values()
        )
        if same_city and match_in_fields(source, RECEIPT):
            eligible.append(key)

    explicit = {key: match_in_fields(sources[key], RIDE) for key in eligible}
    path_index = {}
    for key, hit in explicit.items():
        if hit:
            for phrase, field in phrases(sources[key]):
                path_index.setdefault(phrase, (key, field))
    related = []
    for key in eligible:
        source = sources[key]
        relation = {"id": key, "date": source.day.isoformat(), "media_city": source.city,
                    "receipt_span": match_in_fields(source, RECEIPT)}
        ride = explicit[key]
        route = match_in_fields(source, ROUTE)
        shared = next(((phrase, field, *path_index[phrase]) for phrase, field in phrases(source)
                       if phrase in path_index), None)
        if ride:
            related.append({**relation, "role": "explicit ride receipt", "ride_span": ride})
        elif route:
            related.append({**relation, "role": "receipt with route field", "route_span": route})
        elif shared:
            phrase, field, linked_id, matched_field = shared
            related.append({**relation, "role": "receipt with shared address", "address": phrase,
                            "field": field, "linked_source": linked_id, "linked_field": matched_field})
    return anchor, related

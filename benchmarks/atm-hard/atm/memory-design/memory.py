"""Source-level retrieval views, goal coverage, and extractive evidence packets."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, log
import re

from duorou.retrieval import BM25, rrf, terms

DIRECT_STOP = set(("help find report list remember want know recall retrieval question exact total "
                   "record presentation event during recently used related information recent "
                   "specific memory pieces make use show photo photos image images media video "
                   "taken took visited visit many much which time dates ids answer").split())
PAYMENT = re.compile(r"\b(?:paid|pay|spend|spent|cost|fare|charge|reimburs\w*)\b", re.I)
TIME = re.compile(r"\b(?:when|dates?|time|during|wait|days|nights|start|end|duration|year|stayed)\b", re.I)
PLACE = re.compile(r"\b(?:where|cit(?:y|ies)|countries|location|visit(?:ed)?|hotel|route)\b", re.I)
PROOF = re.compile(r"\b(?:paid in full|amount paid|payment received|debit|contactless|importe|receipt|invoice|refunded)\b", re.I)
PLANNING = re.compile(r"\b(?:book(?:ing|ed)|reservation|itinerary|scheduled|confirmation)\b", re.I)
TEMPORAL = re.compile(r"\b(?:20\d{2}|(?:[01]?\d|2[0-3]):[0-5]\d|arrivals?|departure|check.in|check.out)\b", re.I)
SPLIT = re.compile(r"[^\r\n]+?(?=(?:[.!?](?=\s|$))|\r|\n|$)[.!?]?")
SHORT_FIELDS = {"short_summary","short_caption","ocr_text","detail","caption"}


@dataclass(frozen=True)
class Segment:
    field: str
    start: int
    text: str
    words: frozenset
    roles: frozenset[str]


def roles(text):
    present=set()
    if PROOF.search(text):
        present.add("payment_material")
    if PLANNING.search(text):
        present.add("process")
    if TEMPORAL.search(text):
        present.add("timeline")
    return frozenset(present)


def source_views(sources, tags, facts, clean_ocr):
    views={name:{} for name in ("identity","ocr","amount","date_location")}
    for key,source in sources.items():
        fields=source.fields
        tag=tags[key]
        identity="\n".join(fields.get(f,"") for f in ("short_summary","short_caption"))
        email_subject=re.search(r"(?m)^Subject:[ \t]*(.+)$",fields.get("detail",""))
        if email_subject:
            identity=email_subject[1]+"\n"+identity
        views["identity"][key]=" ".join([identity,*tag["index_terms"]])
        if fields.get("ocr_text"):
            views["ocr"][key]=clean_ocr(fields["ocr_text"])
        valued=list(dict.fromkeys((f.role,f.amount,f.currency) for f in facts.get(key,[])
                                  if f.role!="unclassified"))[:12]
        if valued:
            views["amount"][key]=" ".join(
                [identity,*(" ".join(item) for item in valued)])
        t=source.day
        views["date_location"][key]=" ".join([
            t.isoformat(),f"{t.strftime('%B')} {t.day} {t.year}",source.city,source.location,identity,
        ])
    return views


class FieldIndex:
    def __init__(self, views):
        self.indices={name:BM25(items) for name,items in views.items() if items}

    def find(self, phrases, payment):
        paths=[]
        for phrase in phrases:
            for view,index in self.indices.items():
                if view!="amount" or payment:
                    paths.append(index.search(phrase,12))
        scores=rrf(paths)
        return sorted(scores,key=lambda key:(-scores[key],key))[:60]


def plan(question):
    text=question["question"]
    wanted={"direct":2}
    if PAYMENT.search(text) and question["qtype"]=="number":
        wanted.update(payment_material=4,process=1)
    if TIME.search(text):
        wanted["timeline"]=3
    if PLACE.search(text):
        wanted["place"]=2
    if question["qtype"]=="list_recall":
        wanted["direct"]=6
    return wanted


def frame(source):
    chunks=[]
    for field,text in source.fields.items():
        if field not in SHORT_FIELDS:
            continue
        seen=set()
        for hit in SPLIT.finditer(text):
            value=hit.group()
            if not value.strip():
                continue
            lead=len(value)-len(value.lstrip())
            value=value.strip()
            if value.casefold() in seen:
                continue
            seen.add(value.casefold())
            start=hit.start()+lead
            for offset in range(0,len(value),560):
                part=value[offset:offset+560]
                chunks.append(Segment(field,start+offset,part,frozenset(terms(part)),roles(part)))
    return chunks


def project(source, tag, chunks, required, query, limit):
    metadata=f"Recorded time ({tag['recorded_time_kind']}): {tag['recorded_time'] or 'unknown'}\n"
    if source.city:
        metadata+=f"Recorded place: {source.city}\n"
    ranked=[]
    field_bias={"short_summary":2.9,"short_caption":2.9,"detail":1.0,"ocr_text":.7,"caption":.6}
    for seg in chunks:
        hint=len(seg.words&query)*1.2+field_bias[seg.field]
        hint+=sum(2.4 if type_=="payment_material" else 1.3 for type_ in seg.roles if type_ in required)
        if seg.field=="detail" and seg.text.startswith("Subject:"):
            hint+=4
        ranked.append((hint,seg))
    selected=[]
    used=len(metadata.encode())
    per_field=Counter()
    seen=set()
    for _,seg in sorted(ranked,key=lambda row:(-row[0],row[1].field,row[1].start)):
        if seg.text.casefold() in seen or per_field[seg.field]>=9:
            continue
        size=len(f"{seg.field}: {seg.text}\n".encode())
        if used+size>limit:
            continue
        selected.append(seg)
        seen.add(seg.text.casefold())
        used+=size
        per_field[seg.field]+=1
    selected.sort(key=lambda seg:(seg.field,seg.start))
    text=metadata+"".join(f"{seg.field}: {seg.text}\n" for seg in selected)
    spans=[dict(field=s.field,start=s.start,end=s.start+len(s.text),quote=s.text) for s in selected]
    assert all(source.fields[r["field"]][r["start"]:r["end"]]==r["quote"] for r in spans)
    assert len(text.encode())<=limit
    return text,spans


def sigmoid(raw):
    return 1/(1+exp(-max(-30,min(30,float(raw)/3))))


def choose(question, candidates, prior, passages, logits, wanted, sources, catalog_frequency):
    query=set(terms(question["question"]))-DIRECT_STOP
    active=Counter()
    chosen=[]
    top={key:n for n,key in enumerate(prior)}
    chosen_terms=set()
    selected_features=set()
    fields={}

    for key in candidates:
        words=set(terms(passages[key]))
        category=set(roles(passages[key]))
        if query&words:
            category.add("direct")
        if sources[key].city:
            category.add("place")
        distinctive={w for w in words-query-DIRECT_STOP if len(w)>3 and 1<catalog_frequency[w]<250}
        fields[key]=(words&query,category,distinctive)

    def add(key):
        attrs,kinds,new_content=fields[key]
        chosen.append(key)
        chosen_terms.update(attrs)
        active.update(kinds)
        selected_features.update(new_content)

    for key in prior[:3]:
        if key in fields and key not in chosen:
            add(key)
    remaining=set(candidates)-set(chosen)
    while remaining and len(chosen)<10:
        def score(key):
            attrs,kinds,new_content=fields[key]
            prior_bonus=.35/(1+top[key]/12) if key in top else 0
            relevance=sigmoid(logits[key])
            coverage=sum(.10 for role in kinds if role in wanted and active[role]<wanted[role])
            novel=.065*min(3,len(attrs-chosen_terms))
            if question["qtype"]=="list_recall" and attrs:
                idf=sum(log(1+100/catalog_frequency[w]) for w in sorted(new_content-selected_features)[:8])
                novel+=min(.16,idf*.01)
            return relevance+prior_bonus+coverage+novel
        key=min(remaining,key=lambda k:(-score(k),k))
        add(key)
        remaining.remove(key)
    return chosen,active

"""Bounded retrieval tools and immutable, quoted findings for the Hard experiment."""
from collections import defaultdict
import json
import re

PUBLIC_SHA256 = {
    "emails.json":"5c82c38e7f18923a9eba3af321663a8a5bd70e94e0d5e59e987141c71219a8af",
    "image_batch_results.json":"7204e8b4ab1a0fea97f2003058213742d47589e09636941f8d83b2e77c33e0a1",
    "video_batch_results.json":"88ed14ce32afde1a4d54d5b77a0e7f84305b543186307be6c9bc34119342727a",
    "atm-bench-hard.json":"acd35f2a172a9741d970d2cf21184ff0af8d79a8bf59967fc8aa33d619f6af4a",
    "atm-bench.json":"ab6eaa9df62fb4162e0f5eecd98768a7e3ae721e32d2db2cf227ff41e3295762",
}

MAX_ROUNDS=3
PER_ROUND=18
ROUND_PROMPT="""Review only the open requirements and the newly supplied public source previews.
Previous findings stay recorded and must not be re-verified. For a new finding, cite the exact
source ID and copy a verbatim short quote. Complete can be true only for a fully established
single requirement. Finding one example in a list or one charge in a total does not complete it.
Keep schedule, capture time, actual event time and proven payment separate.
Propose up to three targeted lookups for unresolved requirements. Choose identity for names,
ocr for written content, amount for explicit prices or payment clues, date_location for observed
time/place, full for text, or neighbors for candidate observations near a shown media source.
For neighbors quote the shown source and explain scope in the query if useful. For lexical
lookups use terms grounded in the question or facts; keep an alternative open when times conflict.
Use zero lookups if there is no grounded unresolved lead. Return findings and search actions only."""
ROUND_FIELDS={
    "findings":{"type":"array","items":{"type":"object","additionalProperties":False,
        "required":["slot_id","fact","source_id","quote","complete"],"properties":{
          **{k:{"type":"string"} for k in ("slot_id","fact","source_id","quote")},
          "complete":{"type":"boolean"},
        }}},
    "searches":{"type":"array","items":{"type":"object","additionalProperties":False,
        "required":["slot_id","view","query","anchor_id","anchor_quote"],"properties":{
          **{k:{"type":"string"} for k in ("slot_id","query","anchor_id","anchor_quote")},
          "view":{"type":"string","enum":["identity","ocr","amount","date_location","full","neighbors"]},
        }}},
}


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def union(*groups):
    return list(dict.fromkeys(key for values in groups for key in values))


def prefix(text,size):
    return text.encode()[:size].decode("utf-8",errors="ignore")


def round_robin(paths, excluded, limit):
    queues=[iter(row) for row in paths]
    seen=set(excluded)
    output=[]
    while queues and len(output)<limit:
        for queue in list(queues):
            item=next((x for x in queue if x not in seen),None)
            if item is None:
                queues.remove(queue)
                continue
            output.append(item)
            seen.add(item)
            if len(output)>=limit:
                break
    return output


def slots_from(prior, question):
    requirements=list(dict.fromkeys(item["requirement"].strip() for item in prior
                                   if item["requirement"].strip()))[:6] or [question["question"]]
    return {str(n):{"id":str(n),"requirement":req,
                    "aggregate":question["qtype"]=="list_recall" or bool(re.search(
                        r"\b(?:all|each|total|every|list|multiple|all rides)\b",req,re.I))}
            for n,req in enumerate(requirements,1)}


def neighborhood(graph):
    graph_edges=defaultdict(list)
    for edge in read_rows(graph/"edges.jsonl"):
        if edge["relation"]=="next_nearby_observation":
            a,b=(edge[field].removeprefix("observation:") for field in ("from","to"))
            graph_edges[a].append(b)
            graph_edges[b].append(a)
    return graph_edges


def neighbors_of(src, edges):
    nearby=edges.get(src,[])
    return [item for item in union(nearby,*(edges.get(k,[]) for k in nearby)) if item!=src]


def accept_findings(proposed, state, shown, has_quote):
    known={(row["slot_id"],row["source_id"],row["quote"]) for row in state["ledger"]}
    accepted=[]
    for row in proposed[:8]:
        sid,src,quote=(row[k] for k in ("slot_id","source_id","quote"))
        token=(sid,src,quote)
        if sid not in state["open"] or token in known or not row["fact"].strip():
            continue
        if not has_quote(shown.get(src,""),quote):
            continue
        closed=row["complete"] and not state["open"][sid]["aggregate"]
        accepted.append({**row,"closed":closed})
        known.add(token)
        if closed:
            del state["open"][sid]
    state["ledger"].extend(accepted)
    return accepted


def execute(proposed, state, shown, indices, edges, has_quote):
    lists=[]
    action_log=[]
    for action in proposed[:3]:
        sid,view,anchor=(action[k] for k in ("slot_id","view","anchor_id"))
        if sid not in state["open"]:
            continue
        query=" ".join(action["query"].split())[:200]
        marker=(view,anchor if view=="neighbors" else query.casefold())
        if marker in state["history"]:
            continue
        if view=="neighbors":
            if not has_quote(shown.get(anchor,""),action["anchor_quote"]):
                continue
            retrieved=neighbors_of(anchor,edges)
        elif view in indices and query:
            retrieved=indices[view].search(query,22)
        else:
            continue
        state["history"].add(marker)
        available=[src for src in retrieved if src not in state["shown"]][:8]
        lists.append(available)
        action_log.append({**action,"query":query,"retrieved":available})
    return round_robin(lists,state["shown"],PER_ROUND),action_log

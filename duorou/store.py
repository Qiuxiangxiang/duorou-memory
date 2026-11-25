"""One SQLite database; one scoped service. Identity is deployment-bound, not a tool argument."""
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
import hashlib
import json
import sqlite3
import time
import uuid

from .retrieval import Index

ROOT = Path(__file__).resolve().parents[1]


def now():
    return time.time_ns() // 1000


def uid():
    return uuid.uuid4().hex


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Memory:
    def __init__(self, path, principal, *, embed=None, extractor=None, token_count=None, cache_scopes=16):
        if not principal or cache_scopes < 1:
            raise ValueError('principal and positive cache size required')
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='scopes'").fetchone():
            self.db.executescript((ROOT/'schema/schema.sql').read_text())
            self.db.execute('PRAGMA user_version=2')
        if self.db.execute('PRAGMA user_version').fetchone()[0] != 2:
            raise ValueError('unsupported schema version')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS retired_sources (
            scope_id TEXT NOT NULL, source_id TEXT NOT NULL, replacement_id TEXT NOT NULL,
            retired_at_us INTEGER NOT NULL, PRIMARY KEY(scope_id,source_id),
            FOREIGN KEY(scope_id,source_id) REFERENCES sources(scope_id,source_id) ON DELETE CASCADE
        ) STRICT;
        INSERT INTO tag_definitions SELECT 'memory','topic','text',NULL,'Explicit topic or retrieval hint' WHERE NOT EXISTS (SELECT 1 FROM tag_definitions WHERE namespace='memory' AND tag_key='topic');''')
        self.principal, self.embed, self.extractor, self.token_count = principal, embed, extractor, token_count
        self.lock, self.cache, self.cache_scopes = RLock(), OrderedDict(), cache_scopes
        self.version = self.db.execute('PRAGMA data_version').fetchone()[0]

    def close(self):
        with self.lock:
            self.cache.clear()
            self.db.close()

    @contextmanager
    def tx(self, write=False):
        with self.lock:
            version = self.db.execute('PRAGMA data_version').fetchone()[0]
            if self.version != version:
                self.cache.clear()
                self.version = version
            self.db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            finally:
                after = self.db.execute('PRAGMA data_version').fetchone()[0]
                if write or after != version:
                    self.cache.clear()
                self.version = after

    def _allow(self, scope, write=False):
        permissions = ('write','admin') if write else ('read','admin')
        if not self.db.execute('SELECT 1 FROM scope_grants WHERE scope_id=? AND principal_id=? AND permission IN (?,?)',
                               (scope,self.principal,*permissions)).fetchone():
            raise PermissionError('scope access denied')

    def provision(self, scope, principal=None, permission='admin'):
        """Offline operator only; never exposed to MCP. File access is the trust boundary."""
        principal = principal or self.principal
        with self.tx(True):
            self.db.execute('INSERT OR IGNORE INTO scopes VALUES (?,?)',(scope,scope))
            self.db.execute("INSERT OR IGNORE INTO principals VALUES (?,'agent')",(principal,))
            self.db.execute('INSERT OR IGNORE INTO scope_grants VALUES (?,?,?)',(scope,principal,permission))

    def _entity(self, scope, entity_id, label, kind='person'):
        old = self.db.execute('SELECT label,kind FROM entities WHERE scope_id=? AND entity_id=?',(scope,entity_id)).fetchone()
        if old and (old['label'] != label or old['kind'] != kind):
            raise ValueError('entity ID already has a different identity; use a new ID or an explicit alias')
        self.db.execute("INSERT OR IGNORE INTO entities VALUES (?,?,?,?,'source_bound')",(scope,entity_id,kind,label))

    def _put(self, scope, source_id, text, *, speaker_id=None, speaker=None, session=None,
             timestamp=None, metadata=None, tags=(), source_system='duorou', source_kind='message'):
        if not isinstance(text,str) or not text.strip() or '\x00' in text or not source_id:
            raise ValueError('nonempty source ID and nonempty NUL-free text required')
        if (speaker_id is None) != (speaker is None):
            raise ValueError('speaker ID and label must be supplied together')
        if metadata is not None and not isinstance(metadata,dict):
            raise ValueError('metadata must be an object')
        if not isinstance(tags,(list,tuple)) or not all(isinstance(t,str) and t.strip() for t in tags):
            raise ValueError('tags must be nonempty strings')
        meta = dict(metadata or {}, speaker=speaker, session_key=session)
        meta['topics'] = sorted(set(tags))
        old = self.db.execute('SELECT content,speaker_entity_id,source_time_raw,metadata_json,source_system,source_kind FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
        if old:
            if tuple(old) != (text,speaker_id,timestamp,packed(meta),source_system,source_kind):
                raise ValueError('source ID conflict; use update with a new ID')
            return source_id
        if speaker_id:
            self._entity(scope,speaker_id,speaker)
        self.db.execute('''INSERT INTO sources(scope_id,source_id,source_system,source_kind,external_key,speaker_entity_id,content,content_sha256,recorded_at_us,source_time_raw,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',(scope,source_id,source_system,source_kind,source_id,speaker_id,text,
            hashlib.sha256(text.encode()).hexdigest(),now(),timestamp,packed(meta)))
        span = source_id+':full'
        self.db.execute('INSERT INTO evidence_spans VALUES (?,?,?,0,?)',(scope,span,source_id,len(text)))
        if session:
            self.db.execute('''INSERT OR IGNORE INTO events(scope_id,event_id,kind,title,boundary_origin,status,time_precision)
                VALUES (?,?,'source_session',?,'source_metadata','provisional','unknown')''',(scope,session,session))
            pos = self.db.execute('SELECT coalesce(max(position)+1,0) FROM event_memberships WHERE scope_id=? AND event_id=?',(scope,session)).fetchone()[0]
            self.db.execute('INSERT INTO event_memberships VALUES (?,?,?,?)',(scope,session,span,pos))
        for tag in meta['topics']:
            self.db.execute('''INSERT INTO tag_annotations(scope_id,annotation_id,namespace,tag_key,value_json,source_id,evidence_span_id,origin,status,extractor_version)
                VALUES (?,?,'memory','topic',?,?,?,'metadata','accepted','1.0')''',(scope,uid(),packed(tag),source_id,span))
        return source_id

    def ingest(self, scope, source_id, text, **kwargs):
        with self.tx(True):
            self._allow(scope,True)
            return self._put(scope,source_id,text,**kwargs)

    def ingest_many(self, scope, records):
        """Batch is atomic; input is source records, not benchmark questions/labels."""
        with self.tx(True):
            self._allow(scope,True)
            return [self._put(scope,**record) for record in records]

    def update(self, scope, source_id, replacement_id, text, **kwargs):
        with self.tx(True):
            self._allow(scope,True)
            if source_id == replacement_id or not self.db.execute('SELECT 1 FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone():
                raise ValueError('existing old ID and distinct new ID required')
            old = self.db.execute('SELECT replacement_id FROM retired_sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
            if old:
                if old[0] != replacement_id:
                    raise ValueError('source is already retired')
                if not self.db.execute('SELECT 1 FROM sources WHERE scope_id=? AND source_id=?',(scope,replacement_id)).fetchone():
                    raise ValueError('replacement was deleted')
                return self._put(scope,replacement_id,text,**kwargs)
            if self.db.execute('SELECT 1 FROM sources WHERE scope_id=? AND source_id=?',(scope,replacement_id)).fetchone():
                raise ValueError('replacement must be a new source')
            self._put(scope,replacement_id,text,**kwargs)
            self.db.execute('INSERT INTO retired_sources VALUES (?,?,?,?)',(scope,source_id,replacement_id,now()))
            return replacement_id

    def delete(self, scope, source_id):
        with self.tx(True):
            self._allow(scope,True)
            # Semantic-event descriptions may have been derived from the deleted
            # source, so remove these derived views rather than retain stale hints.
            self.db.execute('''DELETE FROM events WHERE scope_id=? AND kind='semantic_event' AND event_id IN (
                SELECT m.event_id FROM event_memberships m JOIN evidence_spans p USING(scope_id,span_id)
                WHERE m.scope_id=? AND p.source_id=?) AND NOT EXISTS (SELECT 1 FROM events child WHERE child.scope_id=events.scope_id AND child.parent_event_id=events.event_id)''',(scope,scope,source_id))
            deleted = bool(self.db.execute('DELETE FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).rowcount)
            self.db.execute('''DELETE FROM events WHERE scope_id=? AND kind='source_session' AND NOT EXISTS (
                SELECT 1 FROM event_memberships m WHERE m.scope_id=events.scope_id AND m.event_id=events.event_id)
                AND NOT EXISTS (SELECT 1 FROM events child WHERE child.scope_id=events.scope_id AND child.parent_event_id=events.event_id)''',(scope,))
            return deleted

    def _span(self, scope, source_id, start, end):
        row = self.db.execute('SELECT content FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
        if row is None or not 0<=start<end<=len(row[0]):
            raise ValueError('invalid source/span')
        span = uid()
        self.db.execute('INSERT INTO evidence_spans VALUES (?,?,?,?,?)',(scope,span,source_id,start,end))
        return span,row[0][start:end]

    def mention(self, scope, source_id, start, end, entity_id, label, kind='thing'):
        """Explicit source-grounded binding. Never merge by name; stable ID is required."""
        with self.tx(True):
            self._allow(scope,True)
            self._entity(scope,entity_id,label,kind)
            span,surface = self._span(scope,source_id,start,end)
            self.db.execute("INSERT INTO entity_mentions VALUES (?,?,?,?,?,'accepted','1.0',NULL)",(scope,uid(),span,entity_id,surface))
            return span

    def claim(self, scope, source_id, subject_id, attribution_id, predicate, object_text,
              *, claim_id=None, modality='actual', polarity='affirmative', valid_from_us=None,
              valid_to_us=None, start=0, end=None, status='candidate', epistemic_mode='source_assertion', extractor_version='caller/1', if_missing=False):
        """Store an attributed claim; no inferred truth. Revision closes the previous recorded interval."""
        with self.tx(True):
            self._allow(scope,True)
            row = self.db.execute('SELECT content FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
            if row is None:
                raise ValueError('missing source')
            claim_id = claim_id or uid()
            old = self.db.execute('SELECT version,recorded_from_us,recorded_to_us FROM claims WHERE scope_id=? AND claim_id=? ORDER BY version DESC LIMIT 1',(scope,claim_id)).fetchone()
            if old and if_missing:
                return {'id':claim_id,'version':old['version'],'recorded_from_us':old['recorded_from_us']}
            span,_ = self._span(scope,source_id,start,len(row[0]) if end is None else end)
            recorded = max(now(),old['recorded_from_us']+1 if old else 0)
            version = old['version']+1 if old else 1
            if old and old['recorded_to_us'] is None:
                self.db.execute('UPDATE claims SET recorded_to_us=? WHERE scope_id=? AND claim_id=? AND version=?',(recorded,scope,claim_id,old['version']))
            if old and old['recorded_to_us'] is not None:
                recorded = max(recorded,old['recorded_to_us'])
            self.db.execute('''INSERT INTO claims(scope_id,claim_id,version,subject_entity_id,predicate,object_kind,object_text,attribution_entity_id,polarity,modality,epistemic_mode,status,primary_span_id,valid_from_us,valid_to_us,recorded_from_us,extractor_version)
                VALUES (?,?,?,?,?,'text',?,?,?,?,?,?,?,?,?,?,?)''',
                (scope,claim_id,version,subject_id,predicate,object_text,attribution_id,polarity,modality,epistemic_mode,status,span,valid_from_us,valid_to_us,recorded,extractor_version))
            return {'id':claim_id,'version':version,'recorded_from_us':recorded}

    def intention(self, scope, source_id, owner_id, description, *, intention_id=None, state='candidate', completion_id=None, due_at_us=None):
        with self.tx(True):
            self._allow(scope,True)
            intention_id = intention_id or uid()
            cursor = self.db.execute('''INSERT INTO intentions(scope_id,intention_id,owner_entity_id,evidence_span_id,completion_span_id,description,cue_json,due_at_us,state,origin,extractor_version)
                VALUES (?,?,?,?,?,?, '{}',?,?,'model','1.0')
                ON CONFLICT(scope_id,intention_id) DO UPDATE SET completion_span_id=excluded.completion_span_id,state=excluded.state,due_at_us=excluded.due_at_us
                WHERE intentions.owner_entity_id=excluded.owner_entity_id AND intentions.description=excluded.description AND intentions.evidence_span_id=excluded.evidence_span_id''',
                (scope,intention_id,owner_id,source_id+':full',completion_id+':full' if completion_id else None,description,due_at_us,state))
            if cursor.rowcount != 1:
                raise ValueError('intention identity conflict')
            return intention_id

    def consolidate(self, scope, source_id):
        """Conservative, idempotent extraction of explicit first-person plans/decisions.

        No inferred identity or truth. Other claims/entities are provided by callers.
        The entire raw sentence remains the evidence. Returns newly/existing candidate IDs.
        """
        import re
        with self.tx():
            self._allow(scope,True)
            self._allow(scope)
            row = self.db.execute('SELECT content,speaker_entity_id FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
            if not row or not row['speaker_entity_id']:
                return []
            text, actor = row['content'],row['speaker_entity_id']
        # Narrow grammar avoids guessing general natural-language semantics.
        patterns = [(r'(?im)^\s*I (?:plan|intend) to ([^.!?\n]+)', 'plan', 'planned'),
                    (r'(?m)^\s*我(?:计划|打算)([^。！？\n]+)', 'plan', 'planned'),
                    (r'(?im)^\s*I (?:decided|have decided) to ([^.!?\n]+)', 'decision', 'planned'),
                    (r'(?m)^\s*我(?:已经决定|决定)([^。！？\n]+)', 'decision', 'planned')]
        proposals=[]
        for pattern,predicate,modality in patterns:
            for match in re.finditer(pattern,text):
                obj=match.group(1).strip()
                if obj:
                    proposals.append(dict(start=match.start(),end=match.end(),predicate=predicate,object_text=obj,
                                          modality=modality,polarity='affirmative',extractor_version='rule/1',epistemic_mode='source_assertion'))
        if self.extractor:
            # Explicitly configured callback; receives only this source, not query/gold or other scopes.
            extracted=self.extractor(text)
            if not isinstance(extracted,list) or len(extracted)>32:
                raise ValueError('extractor must return at most 32 candidate objects')
            for item in extracted:
                required={'start','end','predicate','object_text','modality','polarity'}
                if not isinstance(item,dict) or set(item)!=required:
                    raise ValueError('unexpected extractor fields')
                if (type(item['start']) is not int or type(item['end']) is not int or not 0<=item['start']<item['end']<=len(text)
                    or not all(isinstance(item[k],str) and 0<len(item[k])<=2048 for k in ('predicate','object_text'))
                    or item['modality'] not in ('actual','planned','conditional','hypothetical') or item['polarity'] not in ('affirmative','negative')):
                    raise ValueError('invalid extractor proposal')
                proposals.append(dict(item,extractor_version='callback/1',epistemic_mode='model_inference'))
        results=[]
        for item in proposals:
            cid='extract:'+hashlib.sha256(packed([source_id,item]).encode()).hexdigest()
            self.claim(scope,source_id,actor,actor,claim_id=cid,status='candidate',if_missing=True,**item)
            results.append(cid)
        return results

    def relate(self, scope, source_id, from_id, from_version, relation, to_id, to_version):
        with self.tx(True):
            self._allow(scope,True)
            self.db.execute('''INSERT INTO claim_relations VALUES (?,?,?,?,?,?,?,'model','1.0')''',
                (scope,from_id,from_version,relation,to_id,to_version,source_id+':full'))

    def event(self, scope, event_id, title, source_ids):
        """Explicit semantic event, independent of source-session grouping."""
        with self.tx(True):
            self._allow(scope,True)
            if not source_ids or len(set(source_ids))!=len(source_ids):
                raise ValueError('event requires distinct source IDs')
            self.db.execute('''INSERT INTO events(scope_id,event_id,kind,title,boundary_origin,status,time_precision)
                VALUES (?,?,'semantic_event',?,'model','accepted','unknown')''',(scope,event_id,title))
            self.db.executemany('INSERT INTO event_memberships VALUES (?,?,?,?)',
                [(scope,event_id,k+':full',i) for i,k in enumerate(source_ids)])
            return event_id

    def alias(self, scope, source_id, entity_id, alias_text):
        with self.tx(True):
            self._allow(scope,True)
            self.db.execute("INSERT INTO entity_aliases VALUES (?,?,?,?,?,'model','accepted','1.0',NULL)",
                (scope,uid(),entity_id,alias_text,source_id+':full'))

    def pin(self, scope, task, source_ids, budget=4096):
        """A task-specific working set references evidence, not duplicated text."""
        with self.tx(True):
            self._allow(scope,True)
            old = self.db.execute('SELECT owner_principal_id FROM working_sets WHERE scope_id=? AND working_set_id=?',(scope,task)).fetchone()
            if old and old[0]!=self.principal:
                raise PermissionError('working set belongs to a different principal')
            self.db.execute('''INSERT INTO working_sets VALUES (?,?,?,?,?,NULL) ON CONFLICT(scope_id,working_set_id)
                DO UPDATE SET token_budget=excluded.token_budget''',(scope,task,self.principal,task,budget))
            self.db.execute('DELETE FROM working_set_members WHERE scope_id=? AND working_set_id=?',(scope,task))
            self.db.executemany('INSERT INTO working_set_members VALUES (?,?,?,?)',
                [(scope,task,k+':full',i) for i,k in enumerate(source_ids)])
            return task

    def working_set(self, scope, task):
        with self.tx():
            self._allow(scope)
            row = self.db.execute('SELECT token_budget FROM working_sets WHERE scope_id=? AND working_set_id=? AND owner_principal_id=?',(scope,task,self.principal)).fetchone()
            if row is None:
                raise PermissionError('working set not accessible')
            out, remaining = [],row[0]
            for r in self.db.execute('''SELECT s.source_id,s.content FROM working_set_members m JOIN evidence_spans p USING(scope_id,span_id)
                JOIN sources s USING(scope_id,source_id) WHERE m.scope_id=? AND m.working_set_id=? ORDER BY m.position''',(scope,task)):
                cost = self.token_count(r['content']) if self.token_count else len(r['content'].encode())
                if cost<=remaining:
                    out.append(dict(r)); remaining-=cost
            return out

    def facts(self, scope, subject_id=None, valid_at_us=None, known_at_us=None):
        with self.tx():
            self._allow(scope)
            t = now() if known_at_us is None else known_at_us
            return [dict(r) for r in self.db.execute('''SELECT c.*,p.source_id FROM claims c JOIN evidence_spans p ON p.scope_id=c.scope_id AND p.span_id=c.primary_span_id
                WHERE c.scope_id=? AND (? IS NULL OR subject_entity_id=?) AND recorded_from_us<=? AND (recorded_to_us IS NULL OR recorded_to_us>?)
                AND NOT EXISTS (SELECT 1 FROM retired_sources r WHERE r.scope_id=p.scope_id AND r.source_id=p.source_id AND r.retired_at_us<=?)
                AND (? IS NULL OR ((valid_from_us IS NULL OR valid_from_us<=?) AND (valid_to_us IS NULL OR valid_to_us>?)))
                ORDER BY recorded_from_us DESC LIMIT 200''',(scope,subject_id,subject_id,t,t,t,valid_at_us,valid_at_us,valid_at_us))]

    def intentions(self, scope, state=None):
        with self.tx():
            self._allow(scope)
            return [dict(r) for r in self.db.execute('SELECT * FROM intentions WHERE scope_id=? AND (? IS NULL OR state=?) ORDER BY due_at_us,intention_id LIMIT 200',(scope,state,state))]

    def evidence(self, scope, source_id):
        with self.tx():
            self._allow(scope)
            row = self.db.execute('SELECT * FROM sources WHERE scope_id=? AND source_id=?',(scope,source_id)).fetchone()
            return dict(row) if row else None

    def timeline(self, scope, limit=50, before_us=None):
        if not 1<=limit<=200:
            raise ValueError('limit must be 1..200')
        with self.tx():
            self._allow(scope)
            return [dict(r) for r in self.db.execute('SELECT source_id,source_time_raw,recorded_at_us,content FROM sources WHERE scope_id=? AND (? IS NULL OR recorded_at_us<?) ORDER BY recorded_at_us DESC,source_id LIMIT ?',(scope,before_us,before_us,limit))]

    def _index(self, scope, history=False):
        cache_key = (scope, history)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        groups, order, rows = {}, {}, {}
        hints, links, entity_members = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in self.db.execute('''SELECT p.source_id,e.event_id,m.position,e.title FROM event_memberships m JOIN evidence_spans p USING(scope_id,span_id)
            JOIN events e USING(scope_id,event_id) WHERE m.scope_id=? AND e.kind='source_session' AND e.status<>'rejected' ORDER BY e.event_id,m.position''',(scope,)):
            groups.setdefault(r['source_id'],r['event_id'])
            order.setdefault(r['source_id'],r['position'])
        event_members = defaultdict(list)
        for r in self.db.execute('''SELECT p.source_id,e.event_id,e.title FROM event_memberships m JOIN evidence_spans p USING(scope_id,span_id)
            JOIN events e USING(scope_id,event_id) WHERE m.scope_id=? AND e.kind='semantic_event' AND e.status='accepted' ORDER BY e.event_id,m.position''',(scope,)):
            hints[r['source_id']].append(r['title']); event_members[r['event_id']].append(r['source_id'])
        for members in event_members.values():
            members = list(dict.fromkeys(members))
            for i,d in enumerate(members):
                links[d].extend(members[max(0,i-2):i]+members[i+1:i+3])
        for r in self.db.execute("SELECT source_id,value_json FROM tag_annotations WHERE scope_id=? AND status='accepted' AND source_id IS NOT NULL",(scope,)):
            hints[r['source_id']].append(str(json.loads(r['value_json'])))
        for r in self.db.execute('''SELECT p.source_id,e.entity_id,e.label FROM entity_mentions m JOIN evidence_spans p USING(scope_id,span_id)
            JOIN entities e ON e.scope_id=m.scope_id AND e.entity_id=m.entity_id WHERE m.scope_id=? AND m.resolution_state='accepted' ORDER BY p.source_id''',(scope,)):
            hints[r['source_id']].append(r['label']); entity_members[r['entity_id']].append(r['source_id'])
        for r in self.db.execute('''SELECT p.source_id,c.subject_entity_id,c.predicate,c.object_text,c.modality,c.polarity,e.label FROM claims c JOIN evidence_spans p ON p.scope_id=c.scope_id AND p.span_id=c.primary_span_id JOIN entities e ON e.scope_id=c.scope_id AND e.entity_id=c.subject_entity_id
            WHERE c.scope_id=? AND c.recorded_to_us IS NULL AND c.status IN ('candidate','active','disputed')''',(scope,)):
            hints[r['source_id']].append(' '.join(str(r[k] or '') for k in ('label','predicate','object_text','modality','polarity')))
            entity_members[r['subject_entity_id']].append(r['source_id'])
        for r in self.db.execute('''SELECT p.source_id,e.alias_text FROM entity_aliases e JOIN evidence_spans p
            ON p.scope_id=e.scope_id AND p.span_id=e.evidence_span_id WHERE e.scope_id=? AND e.status='accepted' ''',(scope,)):
            hints[r['source_id']].append(r['alias_text'])
        for r in self.db.execute('''SELECT p.source_id a,q.source_id b FROM claim_relations r
            JOIN claims c ON c.scope_id=r.scope_id AND c.claim_id=r.from_claim_id AND c.version=r.from_version
            JOIN claims d ON d.scope_id=r.scope_id AND d.claim_id=r.to_claim_id AND d.version=r.to_version
            JOIN evidence_spans p ON p.scope_id=c.scope_id AND p.span_id=c.primary_span_id
            JOIN evidence_spans q ON q.scope_id=d.scope_id AND q.span_id=d.primary_span_id WHERE r.scope_id=?''',(scope,)):
            links[r['a']].append(r['b']); links[r['b']].append(r['a'])
        for members in entity_members.values():
            members = list(dict.fromkeys(members))
            # Avoid quadratic high-degree hubs; only short adjacent windows.
            for i,d in enumerate(members):
                links[d].extend(members[max(0,i-2):i]+members[i+1:i+3])
        source_rows = list(self.db.execute('''SELECT s.* FROM sources s WHERE scope_id=? AND (? OR NOT EXISTS (
            SELECT 1 FROM retired_sources r WHERE r.scope_id=s.scope_id AND r.source_id=s.source_id))''',(scope,history)))
        source_rows.sort(key=lambda r:(groups.get(r['source_id'],r['source_id']),order.get(r['source_id'],0),r['recorded_at_us'],r['source_id']))
        for r in source_rows:
            key = r['source_id']; meta = json.loads(r['metadata_json'])
            text = r['content']; cap = meta.get('blip_caption')
            # Upstream caption is evidence provided with source, not generated here.
            if isinstance(cap,str) and cap:
                text += '\nImage caption: '+cap
            projection = ' '.join([r['source_time_raw'] or '',str(meta.get('speaker') or ''),text,*hints[key]])
            rows[key] = {'text':text,'key':projection,'result':{'id':key,'text':text,'source_time':r['source_time_raw'],'speaker':meta.get('speaker')}}
        index = Index(rows,groups,{k:list(dict.fromkeys(v)) for k,v in links.items()},self.embed)
        self.cache[cache_key] = index
        if len(self.cache)>self.cache_scopes:
            self.cache.popitem(last=False)
        return index

    def search(self, scope, query, top_k=20, budget=32768, mode='hybrid', history=False):
        if not isinstance(query,str) or not query.strip() or len(query)>4096 or not 1<=top_k<=100 or not 1<=budget<=2_000_000 or mode not in ('bm25','enriched','dense','hybrid'):
            raise ValueError('invalid query, top_k, budget or mode')
        with self.tx():
            self._allow(scope)
            index = self._index(scope,history)
            return index.search(query,top_k,budget,set(index.rows),mode,self.token_count)

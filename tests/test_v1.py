import sqlite3
import tempfile
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor
from duorou import Memory
from duorou.server import register

class V1(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory(); self.path=Path(self.dir.name)/'memory.db'
        self.m=Memory(self.path,'alice'); self.m.provision('project'); self.m.provision('private','bob')
        self.m.ingest('project','a','I plan to replace the Orion controller.',speaker_id='alice1',speaker='Alice',session='s1',tags=['hardware'])
        self.m.ingest('project','b','Frequent communication timeout prompted the change.',session='s2')
    def tearDown(self):
        self.m.close(); self.dir.cleanup()
    def ids(self, q, **kw):
        return [x['id'] for x in self.m.search('project',q,**kw)]
    def test_scope_and_grants(self):
        with self.assertRaises(PermissionError): self.m.search('private','Orion')
        with self.assertRaises(PermissionError): self.m.ingest('private','x','private')
        reader=Memory(self.path,'reader'); self.m.provision('project','reader','read')
        try:
            self.assertTrue(reader.search('project','Orion'))
            with self.assertRaises(PermissionError): reader.delete('project','a')
            with self.m.tx(True): self.m.db.execute("DELETE FROM scope_grants WHERE principal_id='reader'")
            with self.assertRaises(PermissionError): reader.search('project','Orion')
        finally: reader.close()
    def test_idempotence(self):
        self.assertEqual(self.m.ingest('project','a','I plan to replace the Orion controller.',speaker_id='alice1',speaker='Alice',session='s1',tags=['hardware']),'a')
        with self.assertRaises(ValueError): self.m.ingest('project','a','changed')
        self.assertEqual(len(self.m.timeline('project')),2)
    def test_atomic_batch(self):
        with self.assertRaises(ValueError): self.m.ingest_many('project',[{'source_id':'new','text':'ok'},{'source_id':'a','text':'conflict'}])
        self.assertIsNone(self.m.evidence('project','new'))
    def test_tags_and_context(self):
        self.assertIn('a',self.ids('hardware'))
        self.m.ingest('project','c','The selected part costs 17 dollars.',session='s1')
        self.assertIn('c',self.ids('Orion'))
    def test_event_and_relation(self):
        self.m.event('project','e','controller upgrade',['a','b'])
        self.assertIn('b',self.ids('Orion',top_k=5))
        self.m.mention('project','a',22,27,'orion','Orion')
        self.m.alias('project','a','orion','constellation')
        self.assertIn('a',self.ids('constellation'))
    def test_update_and_delete(self):
        self.assertIn('a',self.ids('Orion'))
        self.m.update('project','a','a2','Use Pegasus instead.',session='s1')
        self.assertNotIn('a',self.ids('Orion'))
        self.assertIn('a',self.ids('Orion',history=True))
        self.assertIn('a2',self.ids('Pegasus'))
        self.m.delete('project','a2');self.assertNotIn('a',self.ids('Orion'))
        self.m.delete('project','a');self.assertNotIn('a',self.ids('Orion',history=True))
        self.assertEqual([],list(self.m.db.execute('PRAGMA foreign_key_check')))
    def test_invalidation_other_process(self):
        other=Memory(self.path,'alice')
        try:
            self.assertIn('a',self.ids('Orion'));other.delete('project','a')
            self.assertNotIn('a',self.ids('Orion'))
            other.ingest('project','new','Orion returned.');self.assertIn('new',self.ids('Orion'))
        finally: other.close()
    def test_claim_versions(self):
        old=self.m.claim('project','a','alice1','alice1','plan','Orion',modality='planned',valid_from_us=10,valid_to_us=20)
        new=self.m.claim('project','a','alice1','alice1','plan','Pegasus',claim_id=old['id'],modality='planned',valid_from_us=20)
        self.assertEqual(self.m.facts('project',known_at_us=old['recorded_from_us'])[0]['object_text'],'Orion')
        self.assertEqual(self.m.facts('project')[0]['object_text'],'Pegasus')
        self.assertFalse(self.m.facts('project',valid_at_us=15))
        self.m.relate('project','a',new['id'],new['version'],'supersedes',old['id'],old['version'])
        self.m.delete('project','a');self.assertFalse(self.m.facts('project'))
    def test_consolidation(self):
        a=self.m.consolidate('project','a');self.assertEqual(a,self.m.consolidate('project','a'))
        claims=self.m.facts('project');self.assertEqual(len(claims),1)
        self.assertEqual((claims[0]['modality'],claims[0]['status']),('planned','candidate'))
        self.m.ingest('project','zh','我决定更换电池设备。',speaker_id='alice1',speaker='Alice')
        self.assertTrue(self.m.consolidate('project','zh'));self.assertIn('zh',self.ids('电池设备'))
    def test_intention_completion(self):
        i=self.m.intention('project','a','alice1','replace controller')
        with self.assertRaises(sqlite3.IntegrityError): self.m.intention('project','a','alice1','replace controller',intention_id=i,state='completed')
        self.m.intention('project','a','alice1','replace controller',intention_id=i,state='completed',completion_id='b')
        self.assertEqual(self.m.intentions('project')[0]['state'],'completed')
    def test_working_set(self):
        self.m.pin('project','t',['a','b'],budget=65);self.assertEqual(len(self.m.working_set('project','t')),1)
        self.m.provision('project','other');other=Memory(self.path,'other')
        try:
            with self.assertRaises(PermissionError): other.working_set('project','t')
            with self.assertRaises(PermissionError): other.pin('project','t',['a'])
        finally: other.close()
        self.m.delete('project','a');self.assertEqual([x['source_id'] for x in self.m.working_set('project','t')],['b'])
    def test_budget_and_empty(self):
        self.assertFalse(self.ids('xyzzynomatch'))
        self.assertFalse(self.ids('Orion',budget=1))
        with self.assertRaises(ValueError): self.ids('Orion',mode='dense')
    def test_vector_adapter(self):
        # Deliberate fake encoder checks array path only; not model effectiveness.
        def embed(texts): return [[1,0] if 'Orion' in x or 'space' in x else [0,1] for x in texts]
        other=Memory(self.path,'alice',embed=embed)
        try: self.assertEqual(other.search('project','space',mode='dense')[0]['id'],'a')
        finally: other.close()
    def test_threaded(self):
        with ThreadPoolExecutor(max_workers=8) as ex:
            results=list(ex.map(lambda _:self.ids('Orion'),range(100)))
        self.assertTrue(all(r==results[0] for r in results))
    def test_model_extraction_adapter(self):
        def extract(text): return [dict(start=0,end=len(text),predicate='action',object_text='replace',modality='planned',polarity='affirmative')]
        other=Memory(self.path,'alice',extractor=extract)
        try:
            self.assertEqual(len(other.consolidate('project','a')),2)
            self.assertEqual({x['status'] for x in other.facts('project')},{'candidate'})
            self.assertIn('model_inference',{x['epistemic_mode'] for x in other.facts('project')})
            other.extractor=lambda _: [{'bad':True}]
            with self.assertRaises(ValueError): other.consolidate('project','a')
        finally: other.close()
    def test_derived_event_removal(self):
        self.m.event('project','e','secretuniquetitle',['a','b']); self.assertTrue(self.ids('secretuniquetitle'))
        self.m.delete('project','a');self.assertFalse(self.ids('secretuniquetitle'))
    def test_external_comparison_contract(self):
        from benchmarks.v1_eval import compare_external
        docs=[{'id':'a','scope':'s','text':'a'},{'id':'b','scope':'other','text':'b'}]
        qs=[{'id':'q','scope':'s','query':'a','expected_ids':['a']}]
        summary,_=compare_external(docs,qs,[{'query_id':'q','ranked_ids':['a'],'latency_ms':1}])
        self.assertEqual(summary['all_hit'],1)
        with self.assertRaises(ValueError): compare_external(docs,qs,[{'query_id':'q','ranked_ids':['b'],'latency_ms':1}])
        with self.assertRaises(ValueError): compare_external(docs,qs,[])
    def test_existing_schema_migration(self):
        from examples.sqlite_demo import connect_database, initialize_schema
        legacy=Path(self.dir.name)/'legacy.db'; conn=connect_database(legacy);initialize_schema(conn);conn.close()
        m=Memory(legacy,'alice')
        try:
            m.provision('test');m.ingest('test','x','legacy migration');self.assertTrue(m.search('test','migration'))
        finally: m.close()
    def test_tool_registration(self):
        # Exercise same registered functions without pretending the MCP SDK is installed.
        class Registrar:
            def __init__(self): self.tools={}
            def tool(self):
                def add(f): self.tools[f.__name__]=f;return f
                return add
        r=Registrar();register(r,self.m)
        self.assertEqual(len(r.tools),17)
        self.assertEqual(r.tools['memory_ingest']('project','via-mcp','new moon')['id'],'via-mcp')
        self.assertEqual(r.tools['memory_search']('project','moon')['results'][0]['id'],'via-mcp')
        with self.assertRaises(PermissionError): r.tools['memory_search']('private','moon')
if __name__=='__main__': unittest.main()

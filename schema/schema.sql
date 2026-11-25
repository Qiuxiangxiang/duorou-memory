-- duorou memory data layer, schema v0.2. SQLite 3.37+ with JSON and FTS5.
-- Application MUST enable foreign_keys on EVERY connection and authenticate callers.
-- SQLite has no row-level security. Possession of this database bypasses the demo API.
-- Integer timestamps are Unix microseconds; NULL means unknown, not "now".
-- Evidence is immutable in place. Explicit DELETE is allowed; this is not secure
-- erasure of WAL pages, backups, exports, embeddings, or external caches.
PRAGMA foreign_keys = ON;

CREATE TABLE principals (
    principal_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'agent', 'service'))
) STRICT;

CREATE TABLE scopes (
    scope_id TEXT PRIMARY KEY,
    label TEXT NOT NULL
) STRICT;

CREATE TABLE scope_grants (
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
    principal_id TEXT NOT NULL REFERENCES principals(principal_id) ON DELETE CASCADE,
    permission TEXT NOT NULL CHECK (permission IN ('read', 'write', 'admin')),
    PRIMARY KEY (scope_id, principal_id, permission)
) STRICT;
CREATE INDEX ix_grants_principal ON scope_grants(principal_id, scope_id, permission);

-- Identity is scoped. Labels and aliases deliberately have no uniqueness constraint.
CREATE TABLE entities (
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'agent', 'team', 'organization', 'project', 'place', 'thing')),
    label TEXT NOT NULL,
    identity_state TEXT NOT NULL CHECK (identity_state IN ('unresolved', 'source_bound', 'verified')),
    PRIMARY KEY (scope_id, entity_id)
) STRICT;
CREATE INDEX ix_entities_label ON entities(scope_id, label, identity_state);

CREATE TABLE sources (
    source_rowid INTEGER PRIMARY KEY CHECK (source_rowid > 0),
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    source_system TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    external_key TEXT NOT NULL,
    speaker_entity_id TEXT,
    content TEXT NOT NULL CHECK (instr(content, char(0)) = 0),
    content_sha256 TEXT NOT NULL CHECK (
        length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    recorded_at_us INTEGER NOT NULL,
    source_time_raw TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    UNIQUE (scope_id, source_id),
    UNIQUE (scope_id, source_system, external_key),
    FOREIGN KEY (scope_id, speaker_entity_id) REFERENCES entities(scope_id, entity_id)
) STRICT;
CREATE INDEX ix_sources_speaker_recorded ON sources(scope_id, speaker_entity_id, recorded_at_us, source_id);
CREATE INDEX ix_sources_recorded ON sources(scope_id, recorded_at_us, source_id);

-- REPLACE performs an implicit DELETE and can otherwise bypass UPDATE guards
-- or leave stale FTS entries when recursive delete triggers are disabled.
CREATE TRIGGER sources_no_replace BEFORE INSERT ON sources
WHEN EXISTS (SELECT 1 FROM sources s
    WHERE s.source_rowid = new.source_rowid
       OR (s.scope_id = new.scope_id AND s.source_id = new.source_id)
       OR (s.scope_id = new.scope_id AND s.source_system = new.source_system AND s.external_key = new.external_key))
BEGIN
    SELECT RAISE(ABORT, 'source replacement is forbidden; explicit DELETE is required');
END;
CREATE TRIGGER sources_immutable BEFORE UPDATE ON sources BEGIN
    SELECT RAISE(ABORT, 'source is immutable; append a new source or explicitly delete');
END;

-- Only raw text is indexed. Captions and arbitrary metadata are NOT indexed here.
-- This global FTS index is an implementation convenience, not statistical isolation
-- between scopes. Strict tenant isolation needs per-scope indexes or another backend.
CREATE VIRTUAL TABLE source_fts USING fts5(
    content, content='sources', content_rowid='source_rowid', tokenize='unicode61'
);
CREATE TRIGGER sources_fts_insert AFTER INSERT ON sources BEGIN
    INSERT INTO source_fts(rowid, content) VALUES (new.source_rowid, new.content);
END;
CREATE TRIGGER sources_fts_delete AFTER DELETE ON sources BEGIN
    INSERT INTO source_fts(source_fts, rowid, content)
        VALUES ('delete', old.source_rowid, old.content);
END;

-- Offsets: zero-based Unicode codepoints, [char_start, char_end), not bytes,
-- UTF-16 code units or grapheme clusters. NUL is rejected on sources above.
CREATE TABLE evidence_spans (
    scope_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    char_start INTEGER NOT NULL CHECK (char_start >= 0),
    char_end INTEGER NOT NULL CHECK (char_end > char_start),
    PRIMARY KEY (scope_id, span_id),
    FOREIGN KEY (scope_id, source_id) REFERENCES sources(scope_id, source_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_spans_source ON evidence_spans(scope_id, source_id, char_start, char_end);
CREATE TRIGGER spans_bounds BEFORE INSERT ON evidence_spans BEGIN
    SELECT CASE WHEN new.char_end > (
        SELECT length(content) FROM sources
        WHERE scope_id = new.scope_id AND source_id = new.source_id
    ) THEN RAISE(ABORT, 'span exceeds source Unicode codepoint length') END;
END;
CREATE TRIGGER spans_no_replace BEFORE INSERT ON evidence_spans
WHEN EXISTS (SELECT 1 FROM evidence_spans WHERE scope_id = new.scope_id AND span_id = new.span_id)
BEGIN
    SELECT RAISE(ABORT, 'evidence span replacement is forbidden');
END;
CREATE TRIGGER spans_immutable BEFORE UPDATE ON evidence_spans BEGIN
    SELECT RAISE(ABORT, 'evidence span is immutable');
END;

CREATE TABLE entity_mentions (
    scope_id TEXT NOT NULL,
    mention_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    entity_id TEXT,
    surface TEXT NOT NULL,
    resolution_state TEXT NOT NULL CHECK (resolution_state IN ('unresolved', 'candidate', 'accepted')),
    resolver_version TEXT NOT NULL,
    extraction_confidence REAL CHECK (extraction_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (scope_id, mention_id),
    CHECK (resolution_state <> 'accepted' OR entity_id IS NOT NULL),
    FOREIGN KEY (scope_id, span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, entity_id) REFERENCES entities(scope_id, entity_id)
) STRICT;
CREATE INDEX ix_mentions_entity ON entity_mentions(scope_id, entity_id, resolution_state);
CREATE TRIGGER mentions_surface BEFORE INSERT ON entity_mentions BEGIN
    SELECT CASE WHEN new.surface IS NOT (
        SELECT substr(s.content, p.char_start + 1, p.char_end - p.char_start)
        FROM evidence_spans p JOIN sources s USING (scope_id, source_id)
        WHERE p.scope_id = new.scope_id AND p.span_id = new.span_id
    ) THEN RAISE(ABORT, 'mention surface differs from its evidence span') END;
END;
CREATE TRIGGER mentions_surface_update BEFORE UPDATE ON entity_mentions BEGIN
    SELECT CASE WHEN new.surface IS NOT (
        SELECT substr(s.content, p.char_start + 1, p.char_end - p.char_start)
        FROM evidence_spans p JOIN sources s USING (scope_id, source_id)
        WHERE p.scope_id = new.scope_id AND p.span_id = new.span_id
    ) THEN RAISE(ABORT, 'mention surface differs from its evidence span') END;
END;

CREATE TABLE entity_aliases (
    scope_id TEXT NOT NULL,
    alias_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    alias_text TEXT NOT NULL,
    evidence_span_id TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('human', 'rule', 'model')),
    status TEXT NOT NULL CHECK (status IN ('candidate', 'accepted', 'rejected')),
    extractor_version TEXT NOT NULL,
    extraction_confidence REAL CHECK (extraction_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (scope_id, alias_id),
    FOREIGN KEY (scope_id, entity_id) REFERENCES entities(scope_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, evidence_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_aliases_text ON entity_aliases(scope_id, alias_text, status);

CREATE TABLE events (
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('source_session', 'semantic_event')),
    title TEXT NOT NULL,
    boundary_origin TEXT NOT NULL CHECK (boundary_origin IN ('source_metadata', 'rule', 'model', 'human')),
    status TEXT NOT NULL CHECK (status IN ('provisional', 'accepted', 'rejected')),
    session_ordinal INTEGER CHECK (session_ordinal >= 1),
    time_start_us INTEGER,
    time_end_us INTEGER,
    time_precision TEXT NOT NULL CHECK (time_precision IN ('unknown', 'source_local_unspecified', 'instant', 'interval')),
    source_time_raw TEXT,
    parent_event_id TEXT,
    PRIMARY KEY (scope_id, event_id),
    CHECK (time_end_us IS NULL OR time_start_us IS NULL OR time_end_us > time_start_us),
    CHECK (kind <> 'source_session' OR boundary_origin = 'source_metadata'),
    CHECK (parent_event_id IS NULL OR parent_event_id <> event_id),
    FOREIGN KEY (scope_id, parent_event_id) REFERENCES events(scope_id, event_id)
) STRICT;
CREATE INDEX ix_events_session ON events(scope_id, session_ordinal, event_id);
CREATE INDEX ix_events_time ON events(scope_id, time_start_us, time_end_us);
CREATE TABLE event_memberships (
    scope_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (scope_id, event_id, span_id),
    UNIQUE (scope_id, event_id, position),
    FOREIGN KEY (scope_id, event_id) REFERENCES events(scope_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_memberships_span ON event_memberships(scope_id, span_id, event_id);

-- A claim is an attributed assertion, NEVER an automatic guarantee of truth.
-- valid_* is when the assertion applies in the represented world (if known).
-- recorded_* is when this exact version is visible to the memory system.
-- Both intervals are [from, to); NULL endpoints represent unknown/open bounds.
CREATE TABLE claims (
    scope_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    subject_entity_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_kind TEXT NOT NULL CHECK (object_kind IN ('text', 'number', 'boolean', 'entity', 'json')),
    object_text TEXT,
    object_number REAL,
    object_boolean INTEGER CHECK (object_boolean IN (0, 1)),
    object_entity_id TEXT,
    object_json TEXT CHECK (object_json IS NULL OR json_valid(object_json)),
    attribution_entity_id TEXT NOT NULL,
    polarity TEXT NOT NULL CHECK (polarity IN ('affirmative', 'negative')),
    modality TEXT NOT NULL CHECK (modality IN ('actual', 'hypothetical', 'conditional', 'planned')),
    epistemic_mode TEXT NOT NULL CHECK (epistemic_mode IN ('source_assertion', 'tool_observation', 'model_inference', 'quotation')),
    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'disputed', 'retracted', 'superseded')),
    primary_span_id TEXT NOT NULL,
    valid_from_us INTEGER,
    valid_to_us INTEGER,
    recorded_from_us INTEGER NOT NULL,
    recorded_to_us INTEGER,
    extractor_version TEXT NOT NULL,
    extraction_confidence REAL CHECK (extraction_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (scope_id, claim_id, version),
    CHECK (
        (object_kind = 'text' AND object_text IS NOT NULL AND object_number IS NULL AND object_boolean IS NULL AND object_entity_id IS NULL AND object_json IS NULL) OR
        (object_kind = 'number' AND object_text IS NULL AND object_number IS NOT NULL AND object_boolean IS NULL AND object_entity_id IS NULL AND object_json IS NULL) OR
        (object_kind = 'boolean' AND object_text IS NULL AND object_number IS NULL AND object_boolean IS NOT NULL AND object_entity_id IS NULL AND object_json IS NULL) OR
        (object_kind = 'entity' AND object_text IS NULL AND object_number IS NULL AND object_boolean IS NULL AND object_entity_id IS NOT NULL AND object_json IS NULL) OR
        (object_kind = 'json' AND object_text IS NULL AND object_number IS NULL AND object_boolean IS NULL AND object_entity_id IS NULL AND object_json IS NOT NULL)
    ),
    CHECK (valid_to_us IS NULL OR valid_from_us IS NULL OR valid_to_us > valid_from_us),
    CHECK (recorded_to_us IS NULL OR recorded_to_us > recorded_from_us),
    FOREIGN KEY (scope_id, subject_entity_id) REFERENCES entities(scope_id, entity_id),
    FOREIGN KEY (scope_id, object_entity_id) REFERENCES entities(scope_id, entity_id),
    FOREIGN KEY (scope_id, attribution_entity_id) REFERENCES entities(scope_id, entity_id),
    FOREIGN KEY (scope_id, primary_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE UNIQUE INDEX ix_claim_one_open_version ON claims(scope_id, claim_id) WHERE recorded_to_us IS NULL;
CREATE INDEX ix_claim_subject_valid ON claims(scope_id, subject_entity_id, predicate, modality, valid_from_us, valid_to_us);
CREATE INDEX ix_claim_recorded ON claims(scope_id, recorded_from_us, recorded_to_us);
CREATE INDEX ix_claim_object_entity ON claims(scope_id, object_entity_id, predicate) WHERE object_entity_id IS NOT NULL;
CREATE TRIGGER claims_version_order BEFORE INSERT ON claims BEGIN
    SELECT CASE WHEN new.version <> coalesce((
        SELECT max(version) + 1 FROM claims WHERE scope_id = new.scope_id AND claim_id = new.claim_id
    ), 1) THEN RAISE(ABORT, 'claim versions must append sequentially') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM claims c WHERE c.scope_id = new.scope_id AND c.claim_id = new.claim_id
        AND c.recorded_from_us >= new.recorded_from_us
    ) THEN RAISE(ABORT, 'recorded time must advance with claim version') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM claims c WHERE c.scope_id = new.scope_id AND c.claim_id = new.claim_id
        AND (c.recorded_to_us IS NULL OR c.recorded_to_us > new.recorded_from_us)
        AND (new.recorded_to_us IS NULL OR new.recorded_to_us > c.recorded_from_us)
    ) THEN RAISE(ABORT, 'claim recorded-time versions overlap') END;
END;
-- Only closing an open recorded interval is mutable. Correction/status changes
-- are new versions. Closing and appending must occur in one application transaction.
CREATE TRIGGER claims_payload_immutable BEFORE UPDATE ON claims
WHEN old.recorded_to_us IS NOT NULL OR new.recorded_to_us IS NULL
    OR new.scope_id IS NOT old.scope_id OR new.claim_id IS NOT old.claim_id
    OR new.version IS NOT old.version OR new.subject_entity_id IS NOT old.subject_entity_id
    OR new.predicate IS NOT old.predicate OR new.object_kind IS NOT old.object_kind
    OR new.object_text IS NOT old.object_text OR new.object_number IS NOT old.object_number
    OR new.object_boolean IS NOT old.object_boolean OR new.object_entity_id IS NOT old.object_entity_id
    OR new.object_json IS NOT old.object_json OR new.attribution_entity_id IS NOT old.attribution_entity_id
    OR new.polarity IS NOT old.polarity OR new.modality IS NOT old.modality
    OR new.epistemic_mode IS NOT old.epistemic_mode OR new.status IS NOT old.status
    OR new.primary_span_id IS NOT old.primary_span_id OR new.valid_from_us IS NOT old.valid_from_us
    OR new.valid_to_us IS NOT old.valid_to_us OR new.recorded_from_us IS NOT old.recorded_from_us
    OR new.extractor_version IS NOT old.extractor_version
    OR new.extraction_confidence IS NOT old.extraction_confidence
BEGIN
    SELECT RAISE(ABORT, 'claim payload is immutable; close interval and append a version');
END;

CREATE TABLE claim_evidence (
    scope_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    claim_version INTEGER NOT NULL,
    span_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('supports', 'contradicts', 'context')),
    PRIMARY KEY (scope_id, claim_id, claim_version, span_id, role),
    FOREIGN KEY (scope_id, claim_id, claim_version) REFERENCES claims(scope_id, claim_id, version) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_claim_evidence_span ON claim_evidence(scope_id, span_id);
CREATE TABLE claim_relations (
    scope_id TEXT NOT NULL,
    from_claim_id TEXT NOT NULL,
    from_version INTEGER NOT NULL,
    relation TEXT NOT NULL CHECK (relation IN ('supports', 'contradicts', 'supersedes')),
    to_claim_id TEXT NOT NULL,
    to_version INTEGER NOT NULL,
    evidence_span_id TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('human', 'rule', 'model')),
    extractor_version TEXT NOT NULL,
    PRIMARY KEY (scope_id, from_claim_id, from_version, relation, to_claim_id, to_version),
    CHECK (from_claim_id <> to_claim_id OR from_version <> to_version),
    FOREIGN KEY (scope_id, from_claim_id, from_version) REFERENCES claims(scope_id, claim_id, version) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, to_claim_id, to_version) REFERENCES claims(scope_id, claim_id, version) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, evidence_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_claim_relations_target ON claim_relations(scope_id, to_claim_id, to_version, relation);

-- Prospective-memory candidate only: no scheduler or permission to execute actions.
CREATE TABLE intentions (
    scope_id TEXT NOT NULL,
    intention_id TEXT NOT NULL,
    owner_entity_id TEXT NOT NULL,
    evidence_span_id TEXT NOT NULL,
    completion_span_id TEXT,
    description TEXT NOT NULL,
    cue_json TEXT NOT NULL CHECK (json_valid(cue_json) AND json_type(cue_json) = 'object'),
    due_at_us INTEGER,
    state TEXT NOT NULL CHECK (state IN ('candidate', 'active', 'completed', 'cancelled', 'expired')),
    origin TEXT NOT NULL CHECK (origin IN ('human', 'rule', 'model')),
    extractor_version TEXT NOT NULL,
    extraction_confidence REAL CHECK (extraction_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (scope_id, intention_id),
    CHECK (state <> 'completed' OR completion_span_id IS NOT NULL),
    FOREIGN KEY (scope_id, owner_entity_id) REFERENCES entities(scope_id, entity_id),
    FOREIGN KEY (scope_id, evidence_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, completion_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_intentions_due ON intentions(scope_id, state, due_at_us);

-- Controlled vocabulary; add definitions intentionally, never infer authorization
-- from these semantic annotations. confidence estimates extraction, not truth.
CREATE TABLE tag_definitions (
    namespace TEXT NOT NULL,
    tag_key TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('text', 'number', 'boolean', 'enum')),
    allowed_values_json TEXT,
    description TEXT NOT NULL,
    PRIMARY KEY (namespace, tag_key),
    CHECK ((value_type = 'enum' AND allowed_values_json IS NOT NULL
            AND json_valid(allowed_values_json) AND json_type(allowed_values_json) = 'array')
        OR (value_type <> 'enum' AND allowed_values_json IS NULL))
) STRICT;
CREATE TABLE tag_annotations (
    scope_id TEXT NOT NULL,
    annotation_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    tag_key TEXT NOT NULL,
    value_json TEXT NOT NULL CHECK (json_valid(value_json)),
    source_id TEXT,
    event_id TEXT,
    entity_id TEXT,
    claim_id TEXT,
    claim_version INTEGER,
    intention_id TEXT,
    target_span_id TEXT,
    evidence_span_id TEXT,
    origin TEXT NOT NULL CHECK (origin IN ('metadata', 'human', 'rule', 'model')),
    status TEXT NOT NULL CHECK (status IN ('candidate', 'accepted', 'rejected')),
    extractor_version TEXT NOT NULL,
    extraction_confidence REAL CHECK (extraction_confidence BETWEEN 0 AND 1),
    PRIMARY KEY (scope_id, annotation_id),
    CHECK ((source_id IS NOT NULL) + (event_id IS NOT NULL) + (entity_id IS NOT NULL)
        + (claim_id IS NOT NULL) + (intention_id IS NOT NULL) + (target_span_id IS NOT NULL) = 1),
    CHECK ((claim_id IS NULL) = (claim_version IS NULL)),
    CHECK (origin <> 'model' OR evidence_span_id IS NOT NULL),
    FOREIGN KEY (namespace, tag_key) REFERENCES tag_definitions(namespace, tag_key),
    FOREIGN KEY (scope_id, source_id) REFERENCES sources(scope_id, source_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, event_id) REFERENCES events(scope_id, event_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, entity_id) REFERENCES entities(scope_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, claim_id, claim_version) REFERENCES claims(scope_id, claim_id, version) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, intention_id) REFERENCES intentions(scope_id, intention_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, target_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, evidence_span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;
CREATE INDEX ix_tags_lookup ON tag_annotations(scope_id, namespace, tag_key, value_json, status);
CREATE INDEX ix_tags_source ON tag_annotations(scope_id, source_id) WHERE source_id IS NOT NULL;
CREATE INDEX ix_tags_event ON tag_annotations(scope_id, event_id) WHERE event_id IS NOT NULL;
CREATE INDEX ix_tags_claim ON tag_annotations(scope_id, claim_id, claim_version) WHERE claim_id IS NOT NULL;
CREATE TRIGGER tags_value_insert BEFORE INSERT ON tag_annotations BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM tag_definitions d WHERE d.namespace = new.namespace AND d.tag_key = new.tag_key AND (
            (d.value_type = 'text' AND json_type(new.value_json) <> 'text') OR
            (d.value_type = 'number' AND json_type(new.value_json) NOT IN ('integer', 'real')) OR
            (d.value_type = 'boolean' AND json_type(new.value_json) NOT IN ('true', 'false')) OR
            (d.value_type = 'enum' AND NOT EXISTS (
                SELECT 1 FROM json_each(d.allowed_values_json) e
                WHERE e.type = json_type(new.value_json) AND e.value IS json_extract(new.value_json, '$')
            ))
        )
    ) THEN RAISE(ABORT, 'tag value violates controlled definition') END;
END;
CREATE TRIGGER tags_value_update BEFORE UPDATE ON tag_annotations BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM tag_definitions d WHERE d.namespace = new.namespace AND d.tag_key = new.tag_key AND (
            (d.value_type = 'text' AND json_type(new.value_json) <> 'text') OR
            (d.value_type = 'number' AND json_type(new.value_json) NOT IN ('integer', 'real')) OR
            (d.value_type = 'boolean' AND json_type(new.value_json) NOT IN ('true', 'false')) OR
            (d.value_type = 'enum' AND NOT EXISTS (
                SELECT 1 FROM json_each(d.allowed_values_json) e
                WHERE e.type = json_type(new.value_json) AND e.value IS json_extract(new.value_json, '$')
            ))
        )
    ) THEN RAISE(ABORT, 'tag value violates controlled definition') END;
END;
-- Definitions are append-only so existing annotations cannot silently change type.
CREATE TRIGGER tag_definitions_immutable BEFORE UPDATE ON tag_definitions BEGIN
    SELECT RAISE(ABORT, 'tag definition is immutable; introduce a new key/version');
END;
CREATE TRIGGER tag_definitions_no_replace BEFORE INSERT ON tag_definitions
WHEN EXISTS (SELECT 1 FROM tag_definitions WHERE namespace = new.namespace AND tag_key = new.tag_key)
BEGIN
    SELECT RAISE(ABORT, 'tag definition replacement is forbidden; introduce a new key/version');
END;

-- A working set stores selected evidence references, not a duplicate truth store.
-- Budget enforcement, expiry, ranking and owner authorization belong to application.
CREATE TABLE working_sets (
    scope_id TEXT NOT NULL REFERENCES scopes(scope_id) ON DELETE CASCADE,
    working_set_id TEXT NOT NULL,
    owner_principal_id TEXT NOT NULL REFERENCES principals(principal_id),
    task_key TEXT NOT NULL,
    token_budget INTEGER NOT NULL CHECK (token_budget > 0),
    expires_at_us INTEGER,
    PRIMARY KEY (scope_id, working_set_id)
) STRICT;
CREATE TABLE working_set_members (
    scope_id TEXT NOT NULL,
    working_set_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (scope_id, working_set_id, span_id),
    UNIQUE (scope_id, working_set_id, position),
    FOREIGN KEY (scope_id, working_set_id) REFERENCES working_sets(scope_id, working_set_id) ON DELETE CASCADE,
    FOREIGN KEY (scope_id, span_id) REFERENCES evidence_spans(scope_id, span_id) ON DELETE CASCADE
) STRICT;

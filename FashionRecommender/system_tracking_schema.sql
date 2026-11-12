PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    consent_status TEXT NOT NULL DEFAULT 'anonymous'
        CHECK (consent_status IN ('anonymous', 'consented', 'withdrawn')),
    source TEXT NOT NULL DEFAULT 'real'
        CHECK (source IN ('real', 'synthetic'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    trendiness_score REAL NOT NULL CHECK (trendiness_score BETWEEN 0.0 AND 1.0),
    preferred_gender TEXT,
    preference_source TEXT NOT NULL
        CHECK (preference_source IN ('first_purchase', 'explicit', 'inferred', 'synthetic')),
    computed_at_utc TEXT NOT NULL,
    formula_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    gender_label TEXT,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_images (
    image_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    archive_path TEXT NOT NULL UNIQUE,
    mask_path TEXT,
    is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1))
);

CREATE TABLE IF NOT EXISTS category_similarity_rules (
    rule_version TEXT NOT NULL,
    category_left TEXT NOT NULL,
    category_right TEXT NOT NULL,
    group_left TEXT NOT NULL,
    group_right TEXT NOT NULL,
    similarity_score REAL NOT NULL CHECK (similarity_score BETWEEN 0.0 AND 1.0),
    PRIMARY KEY (rule_version, category_left, category_right)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id TEXT PRIMARY KEY,
    experiment_run_id TEXT,
    model_name TEXT NOT NULL,
    artifact_path TEXT,
    artifact_sha256 TEXT,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'retired')),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scoring_configs (
    scoring_config_id TEXT PRIMARY KEY,
    model_version_id TEXT NOT NULL UNIQUE REFERENCES model_versions(model_version_id),
    category_rule_version TEXT,
    visual_weight REAL NOT NULL,
    popularity_weight REAL NOT NULL,
    category_weight REAL NOT NULL DEFAULT 0.0,
    gender_weight REAL NOT NULL DEFAULT 0.0,
    popularity_exponent REAL NOT NULL CHECK (popularity_exponent > 0),
    popularity_threshold REAL,
    candidate_filter_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_embeddings (
    embedding_id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES product_images(image_id),
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    masked INTEGER NOT NULL CHECK (masked IN (0, 1)),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    artifact_path TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (image_id, model_version_id, masked)
);

CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude REAL NOT NULL CHECK (latitude BETWEEN -90.0 AND 90.0),
    longitude REAL NOT NULL CHECK (longitude BETWEEN -180.0 AND 180.0),
    source TEXT NOT NULL CHECK (source IN ('real', 'synthetic'))
);

CREATE TABLE IF NOT EXISTS product_store_inventory (
    product_id TEXT NOT NULL REFERENCES products(product_id),
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    is_available INTEGER NOT NULL DEFAULT 1 CHECK (is_available IN (0, 1)),
    observed_at_utc TEXT NOT NULL,
    PRIMARY KEY (product_id, store_id)
);

CREATE TABLE IF NOT EXISTS recommendation_requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    model_version_id TEXT NOT NULL REFERENCES model_versions(model_version_id),
    query_image_id TEXT REFERENCES product_images(image_id),
    requested_k INTEGER NOT NULL CHECK (requested_k > 0),
    filters_json TEXT NOT NULL DEFAULT '{}',
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL,
    latency_ms REAL CHECK (latency_ms >= 0)
);

CREATE TABLE IF NOT EXISTS recommendation_items (
    request_id TEXT NOT NULL REFERENCES recommendation_requests(request_id)
        ON DELETE CASCADE,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    rank INTEGER NOT NULL CHECK (rank > 0),
    final_score REAL NOT NULL,
    visual_score REAL,
    popularity_score REAL,
    category_score REAL,
    gender_score REAL,
    distance_km REAL CHECK (distance_km >= 0),
    filter_reason TEXT,
    PRIMARY KEY (request_id, product_id),
    UNIQUE (request_id, rank)
);

CREATE TABLE IF NOT EXISTS interaction_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('impression', 'view', 'click', 'favorite', 'cart', 'purchase', 'dismiss')
    ),
    user_id TEXT NOT NULL REFERENCES users(user_id),
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    request_id TEXT REFERENCES recommendation_requests(request_id),
    position INTEGER CHECK (position > 0),
    occurred_at_utc TEXT NOT NULL,
    received_at_utc TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'real' CHECK (source IN ('real', 'synthetic')),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS trend_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(product_id),
    window_start_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    impression_count INTEGER NOT NULL DEFAULT 0 CHECK (impression_count >= 0),
    click_count INTEGER NOT NULL DEFAULT 0 CHECK (click_count >= 0),
    purchase_count INTEGER NOT NULL DEFAULT 0 CHECK (purchase_count >= 0),
    raw_popularity_count INTEGER NOT NULL DEFAULT 0 CHECK (raw_popularity_count >= 0),
    normalized_popularity REAL NOT NULL CHECK (normalized_popularity BETWEEN 0.0 AND 1.0),
    trend_score REAL NOT NULL,
    formula_version TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (product_id, window_start_utc, window_end_utc, formula_version)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, started_at_utc);
CREATE INDEX IF NOT EXISTS idx_images_product ON product_images(product_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_image ON feature_embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_inventory_store ON product_store_inventory(store_id);
CREATE INDEX IF NOT EXISTS idx_requests_session
    ON recommendation_requests(session_id, created_at_utc);
CREATE INDEX IF NOT EXISTS idx_recommendation_items_product
    ON recommendation_items(product_id);
CREATE INDEX IF NOT EXISTS idx_events_user_time
    ON interaction_events(user_id, occurred_at_utc);
CREATE INDEX IF NOT EXISTS idx_events_request
    ON interaction_events(request_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_product_time
    ON interaction_events(product_id, occurred_at_utc);
CREATE INDEX IF NOT EXISTS idx_trends_window
    ON trend_snapshots(window_start_utc, window_end_utc);

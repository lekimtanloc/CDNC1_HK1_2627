PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    version INTEGER PRIMARY KEY,
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    archive_path TEXT NOT NULL,
    archive_bytes INTEGER NOT NULL CHECK (archive_bytes >= 0),
    archive_sha256 TEXT NOT NULL UNIQUE,
    image_count INTEGER NOT NULL CHECK (image_count >= 0),
    item_count INTEGER NOT NULL CHECK (item_count >= 0),
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_builds (
    build_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    created_at_utc TEXT NOT NULL,
    seed INTEGER NOT NULL,
    train_ratio REAL NOT NULL,
    valid_ratio REAL NOT NULL,
    test_ratio REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('passed', 'failed')),
    elapsed_seconds REAL NOT NULL CHECK (elapsed_seconds >= 0),
    error_count INTEGER NOT NULL CHECK (error_count >= 0),
    warning_count INTEGER NOT NULL CHECK (warning_count >= 0),
    protocol_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    build_id TEXT NOT NULL REFERENCES data_builds(build_id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    category TEXT NOT NULL,
    image_count INTEGER NOT NULL CHECK (image_count > 0),
    PRIMARY KEY (build_id, item_id)
);

CREATE TABLE IF NOT EXISTS images (
    build_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    item_id TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    role TEXT NOT NULL CHECK (role IN ('query', 'gallery')),
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    mask_label INTEGER NOT NULL,
    view_category TEXT NOT NULL,
    PRIMARY KEY (build_id, image_path),
    FOREIGN KEY (build_id, item_id) REFERENCES items(build_id, item_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_ground_truth (
    build_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    query_image_path TEXT NOT NULL,
    relevant_gallery_path TEXT NOT NULL,
    item_id TEXT NOT NULL,
    PRIMARY KEY (build_id, query_image_path, relevant_gallery_path),
    FOREIGN KEY (build_id, query_image_path) REFERENCES images(build_id, image_path),
    FOREIGN KEY (build_id, relevant_gallery_path) REFERENCES images(build_id, image_path),
    FOREIGN KEY (build_id, item_id) REFERENCES items(build_id, item_id),
    CHECK (query_image_path <> relevant_gallery_path)
);

CREATE TABLE IF NOT EXISTS validation_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id TEXT NOT NULL REFERENCES data_builds(build_id) ON DELETE CASCADE,
    severity TEXT NOT NULL CHECK (severity IN ('error', 'warning')),
    message TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_artifacts (
    build_id TEXT NOT NULL REFERENCES data_builds(build_id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    bytes INTEGER NOT NULL CHECK (bytes >= 0),
    sha256 TEXT NOT NULL,
    PRIMARY KEY (build_id, path)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    build_id TEXT NOT NULL REFERENCES data_builds(build_id),
    started_at_utc TEXT NOT NULL,
    completed_at_utc TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    variant TEXT NOT NULL,
    backbone TEXT,
    seed INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    code_revision TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    split TEXT NOT NULL,
    name TEXT NOT NULL,
    k INTEGER NOT NULL DEFAULT 0 CHECK (k >= 0),
    value REAL NOT NULL,
    recorded_at_utc TEXT NOT NULL,
    UNIQUE (run_id, split, name, k)
);

CREATE TABLE IF NOT EXISTS experiment_artifacts (
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    bytes INTEGER CHECK (bytes >= 0),
    sha256 TEXT,
    PRIMARY KEY (run_id, path)
);

CREATE INDEX IF NOT EXISTS idx_items_split ON items(build_id, split);
CREATE INDEX IF NOT EXISTS idx_images_item ON images(build_id, item_id);
CREATE INDEX IF NOT EXISTS idx_ground_truth_query
    ON retrieval_ground_truth(build_id, split, query_image_path);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, split);

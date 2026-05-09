# Facade for loading data features
# Refactored to separate concerns into the loading/ folder

from data_prepare.loading.session_manager import (
    get_session_id,
    ensure_cache_dir,
    cache_path,
    list_cache_files,
    session_id_of,
    cleanup_old_files,
    local_path,
    metadata_path,
    cleaned_csv_path,
    target_path,
    ml_cache_path,
    ml_meta_path,
)

from data_prepare.loading.json_parser import (
    decode_json_bytes,
    parse_json_raw,
    get_nested_fields,
    recommend_for_column,
    generate_previews,
    apply_json_overrides,
    read_json_normalized,
)

from data_prepare.loading.file_reader import (
    coerce_object_columns,
    sanitize_for_parquet,
    read_csv_with_fallback,
    process_cached,
    process_data,
)

from data_prepare.loading.state_saver import (
    save_to_local,
    load_from_local,
    save_target_col,
    load_target_col,
    delete_local,
    save_cleaned_data,
    save_ml_cache,
    delete_ml_cache,
    load_ml_cache,
    save_outlier_bounds,
    load_outlier_bounds,
    save_trans_metadata,
    load_trans_metadata,
)

__all__ = [
    # Session Manager
    "get_session_id", "ensure_cache_dir", "cache_path", "list_cache_files", 
    "session_id_of", "cleanup_old_files", "local_path", "metadata_path", 
    "cleaned_csv_path", "target_path", "ml_cache_path", "ml_meta_path",
    
    # JSON Parser
    "decode_json_bytes", "parse_json_raw", "get_nested_fields", 
    "recommend_for_column", "generate_previews", "apply_json_overrides", 
    "read_json_normalized",
    
    # File Reader
    "coerce_object_columns", "sanitize_for_parquet", "read_csv_with_fallback", 
    "process_cached", "process_data",
    
    # State Saver
    "save_to_local", "load_from_local", "save_target_col", "load_target_col", 
    "delete_local", "save_cleaned_data", "save_ml_cache",    "delete_ml_cache", 
    "load_ml_cache",
    "save_outlier_bounds",
    "load_outlier_bounds",
    "save_trans_metadata",
    "load_trans_metadata"
]

//! Write a `.mlpackage` directory that CoreML can open.
//!
//! Layout, which CoreML requires exactly:
//!
//! ```text
//! Foo.mlpackage/
//!   Manifest.json
//!   Data/com.apple.CoreML/model.mlmodel
//! ```
use crate::proto::core_ml::specification as spec;
use prost::Message;
use std::io::Result as IoResult;
use std::path::Path;

/// Write the package, and the weights blob when the graph referenced one.
pub fn write_mlpackage_with_weights(
    dir: &Path,
    model: &spec::Model,
    weights: Option<Vec<u8>>,
) -> IoResult<()> {
    let data = dir.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data)?;
    std::fs::write(data.join("model.mlmodel"), model.encode_to_vec())?;
    let has_weights = weights.is_some();
    if let Some(w) = weights {
        let wd = data.join("weights");
        std::fs::create_dir_all(&wd)?;
        std::fs::write(wd.join("weight.bin"), w)?;
    }

    let id = uuid::Uuid::new_v4().to_string();
    // The weights entry may only appear when the directory does. Listing a path that is not
    // there fails the whole package with "Failed to read model package", code 3.
    let mut items = serde_json::Map::new();
    items.insert(
        id.clone(),
        serde_json::json!({
            "author": "ptts-coreml",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel",
        }),
    );
    if has_weights {
        items.insert(
            uuid::Uuid::new_v4().to_string(),
            serde_json::json!({
                "author": "ptts-coreml",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights",
            }),
        );
    }
    let manifest = serde_json::json!({
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": items,
        "rootModelIdentifier": id,
    });
    std::fs::write(dir.join("Manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}

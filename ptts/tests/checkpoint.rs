//! File-resolution regressions. Fixtures contain no real model weights.
#![cfg(not(target_arch = "wasm32"))]

use ptts::Error;
use ptts::checkpoint::{Checkpoint, MANIFEST_FILE, ModelManifest, ResolveOptions, is_local_source};
use ptts::synth::Quant;
use ptts::tts_model::TTSConfig;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ptts-checkpoint-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let fixture = Self(std::fs::canonicalize(dir).unwrap());
        fixture.write("config.json", &serde_json::to_vec(&TTSConfig::v202601()).unwrap());
        fixture.write("tokenizer.json", b"{}");
        fixture
    }

    fn write(&self, name: &str, bytes: &[u8]) {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn artifact(&self, name: &str) -> Value {
        let bytes = std::fs::read(self.0.join(name)).unwrap();
        json!({"path": name, "sha256": format!("{:x}", Sha256::digest(bytes))})
    }

    fn manifest(&self) -> Value {
        self.write("weights/model.q8.gguf", b"q8 fixture");
        self.write("voices/freya.safetensors", b"voice fixture");
        json!({
            "schema_version": 1,
            "model_id": "test-candidate",
            "revision": "candidate-1",
            "config": self.artifact("config.json"),
            "tokenizer": self.artifact("tokenizer.json"),
            "weights": {"q8_0": self.artifact("weights/model.q8.gguf")},
            "voices": {"freya": self.artifact("voices/freya.safetensors")},
            "default_voice": "freya",
            "sample_rate": 24000,
            "capabilities": {"languages": ["en"], "voice_cloning": false}
        })
    }

    fn save_manifest(&self, manifest: &Value) {
        self.write(MANIFEST_FILE, &serde_json::to_vec(manifest).unwrap());
    }

    fn q8(&self, path: impl AsRef<Path>) -> ptts::Result<Checkpoint> {
        Checkpoint::resolve(path, ResolveOptions { quant: Quant::Q80, weights: None })
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_directory_and_its_config_resolve_identically() {
    let f = Fixture::new();
    f.write("model.safetensors", b"f32");
    f.write("model.q8.gguf", b"q8");
    f.write("embeddings/alba.safetensors", b"voice");
    let dir = f.q8(&f.0).unwrap();
    let config = f.q8(f.0.join("config.json")).unwrap();
    assert_eq!(dir.weights, config.weights);
    assert_eq!(dir.weights.file_name().unwrap(), "model.q8.gguf");
    assert_eq!(dir.tokenizer, config.tokenizer);
    assert_eq!(dir.voices, config.voices);
    assert_eq!(Checkpoint::open(&f.0).unwrap().weights.file_name().unwrap(), "model.safetensors");
}

#[test]
fn legacy_pocket_files_and_config_fallback_remain_supported() {
    let f = Fixture::new();
    std::fs::remove_file(f.0.join("config.json")).unwrap();
    f.write("tts_b6369a24.safetensors", b"legacy f32");
    let ck = f.q8(&f.0).unwrap();
    assert_eq!(ck.config.mimi.sample_rate, 24000);
    assert_eq!(ck.weights.file_name().unwrap(), "tts_b6369a24.safetensors");
}

#[test]
fn legacy_voice_precedence_is_identical_for_all_callers() {
    let f = Fixture::new();
    f.write("model.q8.gguf", b"q8");
    f.write("voices/default.safetensors", b"first");
    f.write("embeddings/default.safetensors", b"second");
    f.write("default-voice.safetensors", b"third");
    f.write("voices/z.safetensors", b"z");
    let ck = f.q8(&f.0).unwrap();
    assert_eq!(
        ck.voices,
        vec![
            ("default".into(), f.0.join("voices/default.safetensors")),
            ("z".into(), f.0.join("voices/z.safetensors"))
        ]
    );
}

#[test]
fn a_manifest_wins_over_legacy_filename_guesses() {
    let f = Fixture::new();
    let manifest = f.manifest();
    f.save_manifest(&manifest);
    f.write("model.q8.gguf", b"decoy");
    f.write("voices/unrelated.safetensors", b"decoy");
    for input in [&f.0, &f.0.join("config.json"), &f.0.join(MANIFEST_FILE)] {
        let ck = f.q8(input).unwrap();
        assert_eq!(ck.weights, f.0.join("weights/model.q8.gguf"));
        assert_eq!(ck.voices.len(), 1);
        assert_eq!(ck.manifest.unwrap().default_voice.as_deref(), Some("freya"));
    }
}

#[test]
fn manifests_require_the_requested_format_or_f32_source() {
    let f = Fixture::new();
    let mut manifest = f.manifest();
    f.save_manifest(&manifest);
    assert!(matches!(Checkpoint::open(&f.0), Err(Error::Unsupported(_))));
    assert!(matches!(
        Checkpoint::resolve(&f.0, ResolveOptions { quant: Quant::Q4k, weights: None }),
        Err(Error::Unsupported(_))
    ));
    f.write("weights/model.safetensors", b"f32 fixture");
    manifest["weights"]["f32"] = f.artifact("weights/model.safetensors");
    f.save_manifest(&manifest);
    let q4 =
        Checkpoint::resolve(&f.0, ResolveOptions { quant: Quant::Q4k, weights: None }).unwrap();
    assert_eq!(q4.weights, f.0.join("weights/model.safetensors"));
    assert_eq!(f.q8(&f.0).unwrap().weights, f.0.join("weights/model.q8.gguf"));
}

#[test]
fn explicit_weights_must_be_declared_and_format_compatible() {
    let f = Fixture::new();
    f.save_manifest(&f.manifest());
    f.write("model.q8.gguf", b"decoy");
    for (name, quant) in [("model.q8.gguf", Quant::Q80), ("weights/model.q8.gguf", Quant::F32)] {
        assert!(matches!(
            Checkpoint::resolve(&f.0, ResolveOptions { quant, weights: Some(name) }),
            Err(Error::Unsupported(_))
        ));
    }
    assert!(
        Checkpoint::resolve(
            &f.0,
            ResolveOptions { quant: Quant::Q80, weights: Some("weights/model.q8.gguf") }
        )
        .is_ok()
    );
}

#[test]
fn every_selected_artifact_is_hash_checked() {
    for name in
        ["config.json", "tokenizer.json", "weights/model.q8.gguf", "voices/freya.safetensors"]
    {
        let f = Fixture::new();
        f.save_manifest(&f.manifest());
        f.write(name, b"corrupted download");
        let err = f.q8(&f.0).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)), "{err}");
        assert!(err.to_string().contains("SHA-256 mismatch"), "{err}");
        assert!(err.to_string().contains(name), "{err}");
    }
}

#[test]
fn missing_declared_files_do_not_fall_back_to_legacy_artifacts() {
    let f = Fixture::new();
    f.save_manifest(&f.manifest());
    std::fs::remove_file(f.0.join("weights/model.q8.gguf")).unwrap();
    f.write("model.q8.gguf", b"legacy");
    assert!(matches!(f.q8(&f.0), Err(Error::NotFound(_))));
}

#[test]
fn invalid_default_voice_sample_rate_and_schema_fail_before_model_loading() {
    let f = Fixture::new();
    let valid = f.manifest();
    for (field, value) in [
        ("default_voice", json!("missing")),
        ("sample_rate", json!(16000)),
        ("schema_version", json!(2)),
    ] {
        let mut manifest = valid.clone();
        manifest[field] = value;
        f.save_manifest(&manifest);
        assert!(f.q8(&f.0).is_err(), "{field}");
    }
}

#[test]
fn artifact_paths_are_portable_and_reject_traversal_components() {
    let f = Fixture::new();
    let valid = f.manifest();
    for path in [
        "../model.gguf",
        "/tmp/model.gguf",
        "C:/model.gguf",
        "..\\model.gguf",
        "a//model.gguf",
        "./model.gguf",
    ] {
        let mut manifest = valid.clone();
        manifest["weights"]["q8_0"]["path"] = json!(path);
        f.save_manifest(&manifest);
        assert!(
            matches!(ModelManifest::read(f.0.join(MANIFEST_FILE)), Err(Error::InvalidData(_))),
            "{path}"
        );
    }
}

#[test]
fn voice_names_cannot_be_used_as_paths_when_exporting() {
    let f = Fixture::new();
    let valid = f.manifest();
    for name in ["../voice", "/voice", "voice\\other", ".", "..", "voice:", "voice\n", " voice"] {
        let mut manifest = valid.clone();
        manifest["voices"] = json!({name: f.artifact("voices/freya.safetensors")});
        manifest["default_voice"] = json!(name);
        f.save_manifest(&manifest);
        assert!(matches!(f.q8(&f.0), Err(Error::InvalidData(_))), "{name}");
    }
}

#[test]
fn future_source_metadata_requires_a_pinned_revision() {
    let f = Fixture::new();
    let mut manifest = f.manifest();
    manifest["source"] = json!({"repo": "test/private-candidate", "revision": "main"});
    f.save_manifest(&manifest);
    assert!(ModelManifest::read(f.0.join(MANIFEST_FILE)).is_err());
    manifest["source"]["revision"] = json!("a".repeat(40));
    f.save_manifest(&manifest);
    assert!(ModelManifest::read(f.0.join(MANIFEST_FILE)).is_ok());
}

#[test]
fn runtime_compatibility_fails_before_model_loading() {
    let f = Fixture::new();
    let mut manifest = f.manifest();
    manifest["runtime_min_version"] = json!("999.0.0");
    f.save_manifest(&manifest);
    assert!(matches!(f.q8(&f.0), Err(Error::Unsupported(_))));
    manifest["runtime_min_version"] = json!("not-a-version");
    f.save_manifest(&manifest);
    assert!(matches!(f.q8(&f.0), Err(Error::InvalidData(_))));
    manifest["runtime_min_version"] = json!(env!("CARGO_PKG_VERSION"));
    f.save_manifest(&manifest);
    assert!(f.q8(&f.0).is_ok());
}

#[test]
fn missing_local_paths_are_not_mistaken_for_hub_ids() {
    for path in ["./missing", "../missing", "/missing/checkpoint", "missing/config.json"] {
        assert!(is_local_source(Path::new(path)), "{path}");
        assert!(matches!(Checkpoint::open(path), Err(Error::NotFound(_))), "{path}");
    }
    let f = Fixture::new();
    let missing = f.0.join("missing");
    assert!(missing.is_absolute());
    assert!(is_local_source(&missing));
    assert!(matches!(Checkpoint::open(&missing), Err(Error::NotFound(_))));
    assert!(!is_local_source(Path::new("owner/private-model")));
}

#[cfg(windows)]
#[test]
fn windows_drive_rooted_paths_are_local_sources() {
    let path = Path::new(r"\missing\checkpoint");
    assert!(path.has_root());
    assert!(!path.is_absolute());
    assert!(is_local_source(path));
    assert!(matches!(Checkpoint::open(path), Err(Error::NotFound(_))));
}

#[cfg(feature = "hf")]
#[test]
#[ignore = "requires PTTS_TEST_MODEL pointing to a private q8 checkpoint"]
fn private_checkpoint_synthesizes_through_the_public_api() {
    let path = std::env::var("PTTS_TEST_MODEL").expect("set PTTS_TEST_MODEL");
    let ck =
        Checkpoint::resolve(path, ResolveOptions { quant: Quant::Q80, weights: None }).unwrap();
    let mut synth = ck
        .builder(ptts::preprocess::Lang::En)
        .device(ptts::synth::DeviceKind::Cpu)
        .build()
        .unwrap();
    ck.register_voices(&mut synth);
    let audio = synth.say("Hello from Phonon.").unwrap();
    assert_eq!(synth.sample_rate(), ck.config.mimi.sample_rate as u32);
    assert!(audio.len() > synth.sample_rate() as usize / 4);
    assert!(audio.iter().all(|x| x.is_finite()));
    assert!(audio.iter().any(|x| x.abs() > 0.01));
    if let Some(manifest) = &ck.manifest {
        assert_eq!(synth.default_voice(), manifest.default_voice);
    }
}

//! Generate Rust types from Apple's CoreML protobuf schema.
//!
//! The schema is vendored under `proto/`, trimmed to what an ML Program needs, so the build
//! needs no network. `protoc` comes from `protoc-bin-vendored`, so it needs no system install
//! either.
fn main() {
    let dir = std::path::Path::new("proto");
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    prost_build::Config::new()
        .protoc_executable(protoc)
        // One file with the whole module tree, rather than one per protobuf package.
        .include_file("coreml_proto.rs")
        .compile_protos(&[dir.join("Model.proto")], &[dir])
        .expect("protoc failed");
    println!("cargo:rerun-if-changed=proto");
}

// `src/mp3.rs` links the system's libmp3lame, which needs LAME 3.99 or later. Debian's
// libmp3lame-dev installs it where the linker already looks. Homebrew does not: ask pkg-config,
// and failing that (pkg-config missing from the PATH a build runs with, an editor's for one, or
// a LAME built without its `lame.pc`) look in Homebrew's prefixes.
const HOMEBREW_LIBS: [&str; 2] = ["/opt/homebrew/lib", "/usr/local/lib"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    // So a `brew install lame` after a failed build is picked up without a `cargo clean`. Only
    // folders that exist: Cargo reruns the script on every build for a path that does not.
    for dir in HOMEBREW_LIBS.into_iter().filter(|dir| std::path::Path::new(dir).is_dir()) {
        println!("cargo:rerun-if-changed={dir}");
    }
    for dir in pkg_config_dirs().unwrap_or_else(homebrew_dirs) {
        println!("cargo:rustc-link-search=native={dir}");
    }
}

fn pkg_config_dirs() -> Option<Vec<String>> {
    let out =
        std::process::Command::new("pkg-config").args(["--libs-only-L", "lame"]).output().ok()?;
    let flags = String::from_utf8(out.stdout).ok().filter(|_| out.status.success())?;
    Some(flags.split_whitespace().filter_map(|f| f.strip_prefix("-L")).map(String::from).collect())
}

fn homebrew_dirs() -> Vec<String> {
    HOMEBREW_LIBS
        .into_iter()
        .filter(|dir| std::path::Path::new(dir).join("libmp3lame.dylib").exists())
        .map(String::from)
        .collect()
}

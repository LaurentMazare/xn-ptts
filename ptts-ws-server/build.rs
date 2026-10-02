// `src/mp3.rs` links the system's libmp3lame. Debian's libmp3lame-dev installs it where the
// linker already looks. Homebrew does not: ask pkg-config, and failing that (pkg-config is not
// always on the PATH a build runs with, an editor's for one) look in Homebrew's prefixes.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
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
    ["/opt/homebrew/lib", "/usr/local/lib"]
        .into_iter()
        .filter(|dir| std::path::Path::new(dir).join("libmp3lame.dylib").exists())
        .map(String::from)
        .collect()
}

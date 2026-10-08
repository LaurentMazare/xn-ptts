# Releasing ptts

Phonon is the model identity. `ptts` is the Rust, Python, and Swift package name. The npm package is `phonon-tts` because the `ptts` name could not be secured. `xn-ptts` is the repository. The native command and the Python console script are both named `ptts` and have different flags. Use `python -m ptts` to choose the Python CLI explicitly.

## Before the first public release

- Confirm publishing access for crates.io, PyPI, npm, GHCR, and GitHub Releases. Configure PyPI's trusted publisher for `maturin-pub.yml`. Confirm ownership of the existing npm `phonon-tts` package and configure its trusted publisher for `npm-publish.yml`. The current placeholder version is not the runtime release. See [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/).
- Select and license the Phonon checkpoint and voices. Prepare its public HF repository privately at the final repo ID and a fixed revision. Prepare the browser files and exported Core ML model bundle from that same checkpoint.
- Test the installed CLI, Python wheel, packed npm package, Docker image, and Swift package with that private revision. Keep private weights and converted models out of public workflow artifacts and shared Actions caches.
- Verify the advertised devices and CPU requirements. Desktop download requirements are in the [CLI guide](cli.md). Python wheel builds and installation tests are in `maturin-pub.yml`; x86 Linux and Windows wheels use x86-64-v3, so older CPUs need a source build. Check each wheel platform before advertising it.

## Prepare a version

1. Update `workspace.package.version` and the workspace `ptts` dependency version in `Cargo.toml`, then update `Cargo.lock`. Python and npm derive their versions from the workspace. Keep model revisions explicit in the examples and model sources you release.
2. Run the package workflows on the release commit without a tag. Download their artifacts and test them in clean consumer environments. `cli-release.yml` builds the desktop binaries and Swift package; `maturin-pub.yml` builds and tests wheels; `npm-publish.yml` builds the browser package.
3. Run `cargo package -p ptts --features cli` and inspect its contents. The Cargo package includes the `ptts` command behind `cli`. `ptts-coreml` is a checkout-only development dependency for exporting models; consumers do not need it to install the packaged CLI. The other workspace crates implement the Python, browser, server, and native-framework products and are not part of the crates.io installation path.
4. Prepare release notes with API changes, model and voice licenses, supported platforms, artifact sizes, and the tested checkpoint revision. Keep unsupported or unverified combinations clear.

## Publish

A `v<version>` tag publishes the Python wheels, npm package, and versioned Docker image through their existing workflows. Confirm all registry settings and the selected model's release readiness before pushing the tag. The native workflow requires that the tag match the workspace version and attaches the desktop and Swift downloads, checksums, and Homebrew formula. It creates a draft GitHub Release when no release exists, so its notes and assets can be reviewed before publication. Retries verify existing assets and upload missing files. They refuse to change existing contents, keeping versioned URLs and Swift checksums stable. Rebuilding all artifacts may change their bytes; use a new version for changed release assets.

Publish the `ptts` Cargo crate with `cargo publish -p ptts` after the package checks pass. Installing its command from crates.io requires `cargo install ptts --locked --features cli`.

The Swift download consists of:

- `PhononCore.xcframework.zip`, the compiled core for iPhone, simulator, and Apple silicon Mac.
- `ptts-swift-<version>.zip`, the Swift wrapper package with that version's framework URL and exact SHA-256 checksum.

The framework contains code, not model weights. Extract the Swift package and add it to Xcode as a local package, selecting the `ptts` product. Its module is `PhononTTS`. Include the library-product rename from `PhononTTS` to `ptts` in the release notes; existing manifest dependencies must select the new product name. After the framework asset is publicly available, verify a clean consumer can download it and build. A repository-based Swift installation additionally needs the generated release manifest committed at the package root with the correct source path. Do that as a release change using the already-published framework archive, rather than guessing the checksum of a future build.

Publish the prepared Core ML model bundle separately with the selected model, its provenance, and license. App developers should be able to download or bundle it without running the exporter.

Review and publish the GitHub Release once its archives and notes are ready. Make the tested HF repository public and expose the browser and Core ML model download URLs at the coordinated model launch.

## Verify publication

From clean consumer environments, check registry installations, GitHub archive downloads and checksums, the Homebrew formula, Swift's framework download, Docker tags, and anonymous HF access. Synthesize speech with the released checkpoint through each advertised path. An earlier private rehearsal verifies loading and runtime behavior; it does not verify public access or registry publication.

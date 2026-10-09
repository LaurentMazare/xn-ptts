#!/usr/bin/env python3
"""Assemble a Swift package that downloads the exact release framework by checksum."""

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    parser.add_argument("--tag", help="Require this tag to match the workspace version")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    if args.tag and args.tag != f"v{version}":
        parser.error(f"tag {args.tag} does not match workspace version {version}")
    framework = args.framework.resolve(strict=True)
    with zipfile.ZipFile(framework) as archive:
        if "PhononCore.xcframework/Info.plist" not in archive.namelist():
            parser.error("framework zip must contain PhononCore.xcframework at its root")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "PhononCore.xcframework.zip"
    shutil.copy2(framework, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = (ROOT / "ios/PhononTTS/Package.swift").read_text()
    manifest = manifest.replace(
        '.binaryTarget(name: "PhononCore", path: "PhononCore.xcframework")',
        '.binaryTarget(\n'
        '            name: "PhononCore",\n'
        f'            url: "https://github.com/gradium-ai/xn-ptts/releases/download/v{version}/PhononCore.xcframework.zip",\n'
        f'            checksum: "{digest}"\n'
        '        )',
    )
    if f'checksum: "{digest}"' not in manifest:
        parser.error("local Swift manifest no longer has the expected binary target")
    name = f"ptts-swift-{version}"
    package = args.output / f"{name}.zip"
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp) / name
        folder.mkdir()
        (folder / "Package.swift").write_text(manifest)
        shutil.copytree(ROOT / "ios/PhononTTS/Sources", folder / "Sources")
        shutil.copy2(ROOT / "ios/PhononTTS/README.md", folder / "README.md")
        for license_file in ("LICENSE-MIT", "LICENSE-APACHE"):
            shutil.copy2(ROOT / license_file, folder / license_file)
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    archive.write(file, f"{name}/{file.relative_to(folder)}")
    checksums = args.output / "SHA256SUMS-apple"
    checksums.write_text(
        f"{digest}  {destination.name}\n"
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n"
    )
    print(f"{package}: framework checksum {digest}")


if __name__ == "__main__":
    main()

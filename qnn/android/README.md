# Phonon on Android with QNN

An Android library around the optimized QNN engine. It streams audio on Snapdragon's HTP,
with shared inference buffers and reused model state. Android 12 or newer, ARM64 only.
There is no automatic CPU fallback.

The package is not published yet. Its proposed Maven coordinates are `ai.gradium:ptts`.
Model weights are separate from the library and retain their own license.

## Build the package

Maintainers need JDK 17 or newer, Android SDK platform 36, NDK 29.0.14206865, CMake 3.31.6,
Rust, cargo-ndk and the QAIRT 2.50.0 SDK headers. Consumers will only need the Maven package.
The matching Qualcomm runtime comes from `com.qualcomm.qti:qnn-runtime:2.50.0` on Maven Central;
do not copy a different version into the app.

```sh
export ANDROID_HOME=/path/to/android-sdk
export QAIRT_ROOT=/path/to/qairt/2.50.0.260828
cargo install cargo-ndk --locked
cd qnn/android
./gradlew :ptts:assembleRelease
```

The output is `ptts/build/outputs/aar/ptts-release.aar`. Maven publishing also produces the
POM that carries the Qualcomm runtime dependency. A standalone AAR file does not carry that
transitive dependency, so use the local Maven repository when testing distribution:

```sh
./gradlew :ptts:publishReleasePublicationToVerificationRepository
./gradlew :smoke:assembleDebug :smoke:assembleRelease \
  -PpttsRepository="$PWD/ptts/build/repository" -PpttsVersion=<workspace-version>
```

This repository only publishes locally. Maven Central access and release automation are
separate release tasks. No private model files, QAIRT headers or vendor binaries are committed.

## Use it in an application

The intended dependency after publication is:

```kotlin
implementation("ai.gradium:ptts:<version>")
```

Enable native extraction in the app's Gradle file.
QNN's DSP loader needs actual files, including the matching Hexagon skel libraries:

```kotlin
android {
    packaging { jniLibs { useLegacyPackaging = true } }
}
```

The library manifest declares optional access to `libcdsprpc.so`. Devices still need working
QNN HTP support from their firmware. An unsupported device or model produces an error.

Load once on a worker thread, then reuse the model:

```kotlin
val tts = PhononTTS.load(context, File(modelDirectory), lang = "en")
tts.setVoice(tts.voices.first())
val sampleRate = tts.sampleRate
val result = tts.speak("Hello from Phonon.") { pcm ->
    // Deliver mono float PCM to AudioTrack at sampleRate.
    audioSink.write(pcm)
    true // Return false to stop.
}
// tts.stop() can also cancel from another thread.
// Wait for speak to finish before closing.
tts.close()
```

Supported normalization choices are `en`, `fr`, `de`, `es`, `pt` and `none`. The language is
required. Callback exceptions propagate, and the model remains reusable after cancellation
or callback failure. Concurrent generation and closing during generation are rejected.

## Model bundles

The existing QNN export path creates `metadata.json`, a tokenizer, BOS data, voices and compiled
context binaries. These are exported model assets, not raw safetensors or GGUF checkpoints.
Exporters must declare the exact tested SoC, matching the target passed to `package.py`:

```sh
# Run from qnn/export after compiling for the target device.
python bundle.py --soc-model SM8750 --out bundle
```

Every `runtime.context_binaries` entry needs an explicit `soc_models` list:

```json
{
  "Galaxy S25": {"file": "phonon.bin", "soc_models": ["SM8750"]}
}
```

The Android loader matches `Build.SOC_MODEL` exactly, after converting it to uppercase. Only
list targets whose correctness and performance have been verified. Existing test bundles need
this metadata added before using the Android library. The app loads its packaged Rust text
library, even when an older bundle includes a copy. Downloaded models cannot supply executable
code to the app.

Bundles may live in an app's private storage or be copied from application assets. The API
currently accepts a local directory. Download and cache helpers can be added when the public
model layout is settled.

## Verify it

```sh
./gradlew :ptts:testDebugUnitTest :smoke:assembleDebug :smoke:assembleRelease
```

Model-free CI runs the Kotlin lifecycle and callback tests without proprietary SDK headers:

```sh
./gradlew :ptts:testDebugUnitTest -PpttsUnitTestsOnly=true
```

That flag refuses packaging tasks, so it cannot accidentally produce an AAR without native code.

The smoke app exercises normal Android application loading and streaming, including a minified
release build. Push a compatible bundle to its external files directory and launch the app:

```sh
adb install -r smoke/build/outputs/apk/debug/smoke-debug.apk
adb shell mkdir -p /sdcard/Android/data/ai.gradium.phonon.smoke/files/model
adb push /path/to/bundle/. /sdcard/Android/data/ai.gradium.phonon.smoke/files/model/
adb shell am start -n ai.gradium.phonon.smoke/.MainActivity
```

Tap Speak. Repeat using `smoke/build/outputs/apk/release/smoke-release.apk` to check R8 and native
loading. A model-free JNI probe can also be launched in either build:

```sh
adb shell am start -n ai.gradium.phonon.smoke/.MainActivity --ez probe true
adb logcat -d -s PhononSmoke:I
```

A successful probe prints `JNI_PROBE_OK`. It verifies native loading, not NPU execution.
Automated installed-app checks are available with:

```sh
./gradlew :smoke:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.modelDir=/sdcard/Android/data/ai.gradium.phonon.smoke/files/model \
  -Pandroid.testInstrumentationRunnerArguments.lang=en
```

Without `modelDir`, the real-model test is skipped and native packaging/JNI loading is still
checked. A real Snapdragon phone is required to verify NPU execution and performance. Desktop
unit tests and APK builds do not establish that the engine works inside an installed app.

### Test through Qualcomm Device Cloud

Use a mobile interactive session with SSH enabled. Forward the remote ADB server to an unused
local port, then pass that port to each ADB command:

```sh
ssh -i /path/to/key -L 5038:<device-host>:5037 -N sshtunnel@ssh.qdc.qualcomm.com
adb -H 127.0.0.1 -P 5038 devices -l
adb -H 127.0.0.1 -P 5038 shell getprop ro.soc.model
```

Choose a bundle compiled for the reported SoC. Install and push it using the commands above,
adding `-H 127.0.0.1 -P 5038` and `-s <serial>` to select the cloud device. Run installed-app
checks directly after installing `smoke/build/outputs/apk/androidTest/debug/smoke-debug-androidTest.apk`:

```sh
adb -H 127.0.0.1 -P 5038 -s <serial> shell am instrument -w \
  -e modelDir /sdcard/Android/data/ai.gradium.phonon.smoke/files/model -e lang en \
  ai.gradium.phonon.smoke.test/androidx.test.runner.AndroidJUnitRunner
```

Then install the minified release APK and run five utterances without playback. The app logs
first-audio time, total generation time and RTF for each run, and saves the last one as a WAV:

```sh
adb -H 127.0.0.1 -P 5038 -s <serial> shell am force-stop ai.gradium.phonon.smoke
adb -H 127.0.0.1 -P 5038 -s <serial> shell am start \
  -n ai.gradium.phonon.smoke/.MainActivity --ez verify true --es lang en
adb -H 127.0.0.1 -P 5038 -s <serial> logcat -d -s PhononSmoke:I
adb -H 127.0.0.1 -P 5038 -s <serial> pull \
  /sdcard/Android/data/ai.gradium.phonon.smoke/files/phonon-verification.wav
```

Wait for `QNN_VERIFY_OK` before pulling the file. Inspect the WAV locally because QDC does not
stream device audio. These results apply to the tested device and firmware; they do not establish
support for other Snapdragon targets. Compare with the native runner on the same device and bundle
when checking for a performance regression.

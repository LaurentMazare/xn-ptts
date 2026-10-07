# Phonon on Android

Phonon runs on Android on the CPU, through [`ptts-ffi`](../ptts-ffi/include/ptts.h), the same C
interface the iOS package wraps. There is no Android package: you build one native library, copy
one Kotlin file into your app, and call it.

## 1. Build the native library

You need Rust, [cargo-ndk](https://github.com/bbqsrc/cargo-ndk) and the Android NDK.

```sh
cargo install cargo-ndk
export ANDROID_NDK_HOME=/path/to/ndk
./android/build.sh
```

This writes `android/jniLibs/arm64-v8a/libptts_ffi.so` for phones and
`android/jniLibs/x86_64/libptts_ffi.so` for the emulator on an x86 machine. Copy the `jniLibs` folder to your app's `src/main/`.

The arm64 build requires the dotprod and fp16 instructions (ARMv8.2), which nearly every phone
since 2019 has. The kernels are chosen at compile time, so the library crashes on a phone without
them. To support older phones, build with `ARM64_FEATURES= ./android/build.sh`. That build runs
everywhere but is slower.

## 2. Add the Kotlin wrapper

Copy [`PhononTTS.kt`](PhononTTS.kt) into your app and change its `package` line. It calls the
library through [JNA](https://github.com/java-native-access/jna):

```kotlin
dependencies {
    implementation("net.java.dev.jna:jna:5.17.0@aar")
}
```

If your release build uses R8 (`isMinifyEnabled = true`), add these lines to
`proguard-rules.pro`. Without them, the release build fails when it loads the library.

```
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-dontwarn java.awt.**
```

## 3. Get a model

`PhononTTS` loads a folder holding `tokenizer.json`, the weights, and voices in `embeddings/` or
`voices/`. Kyutai's Pocket TTS checkpoint, with weights quantized to q8 (147 MB), is these files:

| File in the folder | URL |
|---|---|
| `tts_b6369a24.gguf` | https://huggingface.co/lmz/pocket-tts-without-voice-cloning-q8/resolve/c2d23606a738c5afb5e24e44f9d2f5d6af1b4528/tts_b6369a24.gguf |
| `tokenizer.json` | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/8843db76457a91db32077edf8dfcd1c0e3e755fd/tokenizer.json |
| `embeddings/alba.safetensors` | https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/8843db76457a91db32077edf8dfcd1c0e3e755fd/embeddings/alba.safetensors |

The other voices are `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine` and `azelma`, at the
same path. Download the files into the app's files directory on first run, rather than shipping
them in the APK.

## 4. Speak

```kotlin
// The number of CPU threads; set it before creating the first PhononTTS. The default is one
// per core, which is too many on phones that mix fast and slow cores. Measure on your phones.
android.system.Os.setenv("RAYON_NUM_THREADS", "2", true)

thread {
    val track = AudioTrack.Builder()
        .setAudioFormat(
            AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(24000)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build(),
        )
        .setTransferMode(AudioTrack.MODE_STREAM)
        .build()
    track.play()
    PhononTTS(File(filesDir, "pocket-tts").path, "en").use { tts ->
        tts.setVoice("alba")
        tts.speak("Hello from Phonon.") { pcm ->
            track.write(pcm, 0, pcm.size, AudioTrack.WRITE_BLOCKING)
            true // false stops early
        }
    }
    track.release()
}
```

Loading is the slow part, so an app should keep one `PhononTTS` open rather than load one per
sentence as this example does. Its calls block: make them off the main thread, and from one
thread at a time. Generation runs on its own threads, so a callback that waits for playback, as
above, does not slow it down. `lang` is required: `en`, `fr`, `de`, `es` and `pt` normalize
numbers and symbols for that language, and `none` reads the text as written.

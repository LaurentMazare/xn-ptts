.PHONY: build profiling test clean

WASM_PACK_ARGS = --target web --no-pack --out-dir pkg/wasm --out-name phonon_tts

# The `phonon-tts` npm package, ready to publish from pkg/.
build:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --release
	node scripts/pack.mjs pkg

# Same, without wasm-opt, keeping names for the browser profiler.
profiling:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --profiling
	node scripts/pack.mjs pkg

# Tests for the JS wrapper that need no browser and no model. Needs Node 22.7 or
# later, for module-syntax detection on the `.js` modules under js/.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg

.PHONY: build profiling test clean check-wasm-opt

WASM_PACK_ARGS = --target web --no-pack --out-dir pkg/wasm --out-name phonon_tts

# The `phonon-tts` npm package, ready to publish from pkg/.
build: check-wasm-opt
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --release
	node scripts/pack.mjs pkg

# Same, without wasm-opt, keeping names for the browser profiler.
profiling:
	rm -rf pkg
	wasm-pack build $(WASM_PACK_ARGS) --profiling
	node scripts/pack.mjs pkg

# wasm-pack runs the first wasm-opt on PATH, and downloads binaryen 117 when there is none.
# That one aborts on this module ("UNREACHABLE executed" in its interpreter), so the release
# build needs a newer binaryen on PATH. 133 is the version this was checked with.
WASM_OPT_MIN = 133
check-wasm-opt:
	@v=$$(wasm-opt --version 2>/dev/null | sed -n 's/^wasm-opt version \([0-9]*\).*/\1/p'); \
	if [ -z "$$v" ] || [ "$$v" -lt $(WASM_OPT_MIN) ]; then \
	  echo "make build needs binaryen's wasm-opt $(WASM_OPT_MIN) or later on PATH (found: $${v:-none})."; \
	  echo "Install it with 'brew install binaryen', or from https://github.com/WebAssembly/binaryen/releases."; \
	  exit 1; \
	fi

# Tests for the JS wrapper that need no browser and no model. Needs Node 22.7 or
# later, for module-syntax detection on the `.js` modules under js/.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg

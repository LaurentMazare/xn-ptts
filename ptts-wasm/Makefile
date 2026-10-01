.PHONY: build profiling demo serve test clean check-wasm-opt

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
# Releases up to 123 abort on this module ("UNREACHABLE executed" in their interpreter), so
# the release build needs binaryen 124 or later on PATH. CI builds with exactly 124.
WASM_OPT_MIN = 124
check-wasm-opt:
	@v=$$(wasm-opt --version 2>/dev/null | sed -n 's/^wasm-opt version \([0-9]*\).*/\1/p'); \
	if [ -z "$$v" ] || [ "$$v" -lt $(WASM_OPT_MIN) ]; then \
	  echo "make build needs binaryen's wasm-opt $(WASM_OPT_MIN) or later on PATH (found: $${v:-none})."; \
	  echo "Install it with 'brew install binaryen', or from https://github.com/WebAssembly/binaryen/releases."; \
	  echo "For an unoptimized local build without binaryen, run 'make profiling' instead."; \
	  exit 1; \
	fi

# The demo page, using the package exactly as a consumer would, in site/.
demo: build
	rm -rf site
	mkdir -p site
	cp -r pkg site/phonon-tts
	cp www/index.html site/

serve: demo
	python3 -m http.server -d site 8080

# Tests for the JS wrapper that need no browser and no model. Needs Node 22.7 or
# later, for module-syntax detection on the `.js` modules under js/.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg site

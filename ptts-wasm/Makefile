.PHONY: build profiling test clean

build:
	wasm-pack build --target web --release
	cp www/index.html www/worker.js pkg/

profiling:
	wasm-pack build --target web --profiling
	cp www/index.html www/worker.js pkg/

# Tests for the JS wrapper that need no browser and no model.
test:
	node --test js/test/*.test.mjs

clean:
	rm -rf pkg

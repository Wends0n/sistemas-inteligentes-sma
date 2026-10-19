CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS = -Iinclude
LDLIBS = -lm
CORE = src/sma.c src/pfs.c src/rng.c
HEADERS = include/sma.h include/pfs.h include/rng.h
.PHONY: all test experiment plot clean sanitize
all: build/sma_pfs
build:
	mkdir -p build
build/sma_pfs: src/main.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c $(CORE) $(LDLIBS) -o $@
build/test: tests/test.c $(CORE) $(HEADERS) | build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test.c $(CORE) $(LDLIBS) -o $@
test: build/test
	./build/test
experiment: all
	mkdir -p results
	./build/sma_pfs results 30 500 30
plot:
	python3 scripts/analyze.py results
sanitize: | build
	$(CC) $(CPPFLAGS) -std=c11 -g -O1 -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -fno-omit-frame-pointer tests/test.c $(CORE) $(LDLIBS) -o build/test_sanitize
	./build/test_sanitize
clean:
	rm -rf build

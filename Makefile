specs_small := $(wildcard specs/swinv_like/l?/*.json)
specs_big := $(wildcard specs/swinv_like_big/l?/*.json)

swizzle_inventor_tests_small := $(wildcard swizzle_inventor_benchmarks/small_*.rkt)
swizzle_inventor_tests_big := $(wildcard swizzle_inventor_benchmarks/big_*.rkt)

SWIZZLEFLOW_FLAGS ?= -a
build:
	cargo build --release
build-stats:
	cargo build --features stats --release

timings: build ${specs_small}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_small}
timings-all: build ${specs_small} ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big} ${specs_small}

stats: build-stats ${specs_small}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_small}
stats-all: build-stats ${specs_small} ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big} ${specs_small}

swizzle-inventor: swizzle_inventor_benchmarks/benchmark.sh ${swizzle_inventor_tests_small}
	./swizzle_inventor_benchmarks/benchmark.sh ${swizzle_inventor_tests_small}
swizzle-inventor-all: swizzle_inventor_benchmarks/benchmark.sh ${swizzle_inventor_tests_small} ${swizzle_inventor_tests_big}
	./swizzle_inventor_benchmarks/benchmark.sh ${swizzle_inventor_tests_big} ${swizzle_inventor_tests_small}

clean:
	rm -rf target
clean-matrices:
	rm matrices/*

.PHONY: build build-stats clean timings timings-all clean-matrices stats stats-all\
swizzle-inventor swizzle-inventor-all

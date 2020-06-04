specs_small := $(wildcard specs/swinv_like/l?/*.json)
specs_big := $(wildcard specs/swinv_like_big/l?/*.json)

swizzle_inventor_tests_small := $(wildcard swizzle-inventor-benchmarks/small_*.rkt)
swizzle_inventor_tests_big := $(wildcard swizzle-inventor-benchmarks/big_*.rkt)

specs_matmul_large :=\
specs/swinv_like/l1/trove-crc-2.json\
specs/swinv_like/l1/trove-crc-3.json\
specs/swinv_like/l1/trove-crc-4.json\
specs/swinv_like/l1/trove-rcr-4.json\
specs/swinv_like/l1/trove-cr_sum-5.json\
specs/swinv_like/l1/trove-cr_sum-7.json\
specs/swinv_like/l1/trove-crc-5.json\
specs/swinv_like/l1/2d-stencil-5.json

SWIZZLEFLOW_FLAGS ?= -a
build:
	cargo build --release
build-stats:
	cargo build --features stats --release

timings: build ${specs_small}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_small}
timings-big: build ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big}
timings-all: build ${specs_small} ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big} ${specs_small}

stats: build-stats ${specs_small}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_small}
stats-big: build-stats ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big}
stats-all: build-stats ${specs_small} ${specs_big}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_big} ${specs_small}

matmul-large-timings: build ${specs_matmul_large}
	./target/release/swizzleflow ${SWIZZLEFLOW_FLAGS} ${specs_matmul_large}

swizzle-inventor: swizzle-inventor-benchmarks/benchmark.sh ${swizzle_inventor_tests_small}
	./swizzle-inventor-benchmarks/benchmark.sh ${swizzle_inventor_tests_small}
swizzle-inventor-all: swizzle-inventor-benchmarks/benchmark.sh ${swizzle_inventor_tests_small} ${swizzle_inventor_tests_big}
	./swizzle-inventor-benchmarks/benchmark.sh ${swizzle_inventor_tests_big} ${swizzle_inventor_tests_small}

clean:
	rm -rf target
clean-matrices:
	rm matrices/*

.PHONY: build build-stats clean timings timings-all clean-matrices stats stats-all\
swizzle-inventor swizzle-inventor-all

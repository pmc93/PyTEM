CMAKE ?= cmake
BUILD_DIR ?= build
BUILD_TYPE ?= Release

.PHONY: all configure build test run clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTING=ON

build: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/pytem-inversion-gui

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)

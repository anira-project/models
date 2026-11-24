set shell := ["bash", "-cu"]

preset := "debug"

default:
    @just --list

# The dev configure (compile DB for clangd)
setup:
    cmake --preset {{preset}}

# Configure + build
build:
    cmake --preset {{preset}} > /dev/null
    cmake --build --preset {{preset}}

# Build + run every test
test: build
    ctest --preset {{preset}}

# clang-format in place (the tracked sources only, never a .venv)
format:
    git ls-files '*.cpp' '*.h' | xargs clang-format -i

# clang-format check
format-check:
    git ls-files '*.cpp' '*.h' | xargs clang-format --dry-run --Werror

# Every format of every description over consecutive calls, and every engine and backend present
check-streaming WEIGHTS:
    uv run tools/check_streaming.py {{WEIGHTS}}

# Reinstall the tanh-tooling copies (.clang-*, cmake/tanh/) at the pinned version
tooling-install VERSION="v0.3.0":
    curl -fsSL https://raw.githubusercontent.com/tanh-lab/tanh-tooling/{{VERSION}}/install.sh | sh -s -- clang cmake

# Remove every build directory
clean:
    rm -rf build

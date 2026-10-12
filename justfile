set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# List development commands.
default:
    @just --list

# Build all crates and targets.
build:
    cargo build --workspace --all-targets --all-features --locked

# Check compilation without generating binaries.
check:
    cargo check --workspace --all-targets --all-features --locked

# Format Rust sources.
fmt:
    cargo fmt --all

# Verify Rust formatting.
fmt-check:
    cargo fmt --all -- --check

# Treat Clippy warnings as errors.
lint:
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

# Run unit, integration, and documentation tests.
test:
    cargo test --workspace --all-features --locked

# Verify the facade with optional modules disabled and enabled individually.
features:
    cargo check -p eta --no-default-features --locked
    cargo check -p eta --no-default-features --features ml --locked
    cargo check -p eta --no-default-features --features nn --locked
    cargo check -p eta --no-default-features --features rl --locked

# Build API documentation with warnings treated as errors.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked

# Run an example from the eta facade.
example name="tensor":
    cargo run -p eta --example {{ name }} --locked

# Run the local CI checks.
ci: fmt-check lint test features doc

# Enter the reproducible development environment.
dev:
    nix develop

# Validate the Nix flake.
nix-check:
    nix flake check

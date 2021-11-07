#!/bin/sh -ex

export EXAMPLE_NAME="${1}"
cargo build \
    --release \
    --example "${EXAMPLE_NAME}" \
    --target wasm32-unknown-unknown \
    --no-default-features \
    --features bevy_core_pipeline,bevy_gltf2,bevy_sprite2,bevy_render2,bevy_pbr2,bevy_winit,png,x11,jpeg \
&& wasm-bindgen \
    --out-dir web \
    --target web \
    "target/wasm32-unknown-unknown/release/examples/${EXAMPLE_NAME}.wasm" \
&& envsubst < index.html.template > "web/${EXAMPLE_NAME}.html"

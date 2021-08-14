export EXAMPLE_NAME=$1
cargo build --example $1 --target wasm32-unknown-unknown --no-default-features --features bevy_core_pipeline,bevy_gltf2,bevy_sprite2,bevy_render2,bevy_pbr2,bevy_winit,png,x11 --release;
wasm-bindgen --target web --out-dir web --no-typescript target/wasm32-unknown-unknown/release/examples/$1.wasm
envsubst < index.html.template > $1.html


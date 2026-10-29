use common_lingua::{LANGS, Model, encode};
use js_sys::Map;
use wasm_bindgen::prelude::*;

use std::sync::OnceLock;
static INIT: OnceLock<Model> = OnceLock::new();

#[wasm_bindgen]
pub fn load(b: &[u8]) {
    let mut model = Model::load(b).unwrap();
    model.tune();
    INIT.set(model).ok();
}

#[wasm_bindgen]
pub fn detect(s: &str) -> Map {
    let model = INIT.get().unwrap();
    let js_map = Map::new();
    for (k, v) in model.predict(&encode(s)).into_iter().enumerate() {
        js_map.set(&JsValue::from_str(LANGS[k]), &JsValue::from_f64(v as f64));
    }
    js_map
}

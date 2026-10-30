use common_lingua::{LANGS, Model, encode};
use js_sys::Map;
use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

thread_local! {
    static INIT: OnceLock<Model> = OnceLock::new();
}

#[wasm_bindgen]
pub fn load(b: &[u8]) -> String {
    let mut model = Model::load(b).unwrap();
    model.tune();
    let prof = model.profile.to_string();
    INIT.with(|f| f.set(model)).ok();
    prof
}

#[wasm_bindgen]
pub fn detect(s: &str) -> Map {
    let js_map = INIT.with(|f| {
        let model = f.get().unwrap();
        let js_map = Map::new();
        for (k, v) in model.predict(&encode(s)).into_iter().enumerate() {
            js_map.set(&JsValue::from_str(LANGS[k]), &JsValue::from_f64(v as f64));
        }
        js_map
    });
    js_map
}

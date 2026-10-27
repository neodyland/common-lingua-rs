use ::common_lingua::ModelFp32;
use pyo3::prelude::*;

use std::sync::OnceLock;
static INIT: OnceLock<ModelFp32> = OnceLock::new();

/// Fast common lingua inference library
#[pymodule]
mod common_lingua {
    use std::collections::HashMap;

    use super::INIT;
    use ::common_lingua::{LANGS, ModelFp32, encode};
    use pyo3::prelude::*;

    /// detect the string's language
    #[pyfunction]
    fn detect(s: &str) -> HashMap<String, f32> {
        let model = INIT.get_or_init(ModelFp32::load);
        let mut map = HashMap::new();
        for (k, v) in model.predict(&encode(s)).into_iter().enumerate() {
            map.insert(LANGS[k].to_string(), v);
        }
        map
    }
}

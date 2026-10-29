use pyo3::prelude::*;

/// Fast common lingua inference library
#[pymodule]
mod common_lingua {
    use std::collections::HashMap;

    use ::common_lingua::{LANGS, Model as RModel, encode};
    use pyo3::prelude::*;

    #[pyclass]
    struct Model(RModel);

    #[pymethods]
    impl Model {
        #[new]
        fn new() -> Self {
            let mut model =
                RModel::load(include_bytes!("../../common-lingua/src/weight.safetensors")).unwrap();
            model.tune();
            Model(model)
        }

        fn detect(&self, s: &str) -> HashMap<String, f32> {
            let mut map = HashMap::new();
            for (k, v) in self.0.predict(&encode(s)).into_iter().enumerate() {
                map.insert(LANGS[k].to_string(), v);
            }
            map
        }
    }
}

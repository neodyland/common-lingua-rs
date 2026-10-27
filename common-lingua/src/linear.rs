use crate::load_weight;
pub struct NoBiasLinear {
    pub weight: Vec<f32>,
    out: usize,
    inp: usize,
}

impl NoBiasLinear {
    pub fn load(name: &str) -> Self {
        let (weight, shape) = load_weight(&format!("{name}.weight"));
        let out = shape[0];
        let inp = shape[1];
        Self { weight, out, inp }
    }
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.inp;
        let mut out = vec![0.0f32; rows * self.out];
        for row in 0..rows {
            for (o, weights) in self.weight.chunks_exact(self.inp).enumerate() {
                let mut sum = 0_f32;
                for i in 0..self.inp {
                    sum += weights[i] * x[row * self.inp + i];
                }
                out[row * self.out + o] = sum;
            }
        }
        out
    }
}

pub struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
    out: usize,
    inp: usize,
}

impl Linear {
    pub fn load(name: &str) -> Self {
        let (weight, shape) = load_weight(&format!("{name}.weight"));
        let out = shape[0];
        let inp = shape[1];
        let bias = load_weight(&format!("{name}.bias")).0;
        Self {
            weight,
            bias,
            out,
            inp,
        }
    }
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.inp;
        let mut out = vec![0.0f32; rows * self.out];
        for row in 0..rows {
            for (o, (weights, sum)) in self
                .weight
                .chunks_exact(self.inp)
                .zip(&self.bias)
                .enumerate()
            {
                let mut sum = *sum;
                for i in 0..self.inp {
                    sum += weights[i] * x[row * self.inp + i];
                }
                out[row * self.out + o] = sum;
            }
        }
        out
    }
}

pub fn fast_exp(x: f32) -> f32 {
    const LOG2E: f32 = 1.442_695_040_888_963_4;
    const LN2_HI: f32 = 0.693_145_751_953_125;
    const LN2_LO: f32 = 1.428_606_765_330_187e-6;
    const MAGIC: f32 = 12_582_912.0;

    let z = x * LOG2E;
    let n = (z + MAGIC) - MAGIC;
    let r = x - n * LN2_HI - n * LN2_LO;

    let mut p = 1.0 / 5040.0;
    p = r.mul_add(p, 1.0 / 720.0);
    p = r.mul_add(p, 1.0 / 120.0);
    p = r.mul_add(p, 1.0 / 24.0);
    p = r.mul_add(p, 1.0 / 6.0);
    p = r.mul_add(p, 0.5);
    p = r.mul_add(p, 1.0);
    p = r.mul_add(p, 1.0);

    let ni = (n as i32).clamp(-126, 127);
    let scale = f32::from_bits(((ni + 127) as u32) << 23);
    p * scale
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + fast_exp(-x))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x / 2.0f64.sqrt()))) as f32
}

pub fn softmax(values: &mut [f32]) {
    let n = values.len();
    softmax_rows(values, n);
}

pub fn softmax_rows(data: &mut [f32], n: usize) {
    if n == 0 || data.is_empty() {
        return;
    }
    for row in data.chunks_exact_mut(n) {
        let mut max = f32::NEG_INFINITY;
        for value in row.iter() {
            if *value > max {
                max = *value;
            }
        }
        let mut sum = 0.0f32;
        for value in row.iter_mut() {
            let e = fast_exp(*value - max);
            *value = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for value in row.iter_mut() {
            *value *= inv;
        }
    }
}

pub fn layer_norm_into(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    n: usize,
    weight: &[f32],
    bias: &[f32],
) {
    const EPS: f32 = 1e-5;
    for r in 0..rows {
        let base = r * n;
        let row = &src[base..base + n];
        let out = &mut dst[base..base + n];

        let mut sum = 0.0f32;
        for value in row.iter() {
            sum += value;
        }
        let mean = sum / n as f32;

        let mut var = 0.0f32;
        for value in row.iter() {
            let diff = value - mean;
            var = diff.mul_add(diff, var);
        }
        var /= n as f32;
        let inv = 1.0 / (var + EPS).sqrt();

        for i in 0..n {
            out[i] = (row[i] - mean).mul_add(inv * weight[i], bias[i]);
        }
    }
}

pub fn silu_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = silu(*value);
    }
}

pub fn swiglu_inplace(data: &mut [f32], rows: usize, cols: usize) {
    let stride = 2 * cols;
    for r in 0..rows {
        let base = r * stride;
        let (gate, rest) = data[base..base + stride].split_at_mut(cols);
        for (g, u) in gate.iter_mut().zip(rest[..cols].iter()) {
            *g = silu(*g) * *u;
        }
    }
}

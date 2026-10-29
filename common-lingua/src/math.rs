static LOG2E: f32 = 1.0 / std::f32::consts::LN_2;

static LN2_HI: f32 = f32::from_bits(std::f32::consts::LN_2.to_bits() & !0x1F);
static LN2_LO: f32 = std::f32::consts::LN_2 - LN2_HI;
static MAGIC: f32 = 12_582_912.0;

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
pub fn fast_exp(x: f32) -> f32 {
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

pub fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf(x / 2.0f64.sqrt()))) as f32
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

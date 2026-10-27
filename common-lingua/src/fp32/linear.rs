use crate::weights::load_weight;

pub struct NoBiasLinear {
    pub weight_b: Vec<f32>,
    pub out: usize,
}

impl NoBiasLinear {
    pub fn load(name: &str) -> Self {
        let (weight, shape) = load_weight(&format!("{name}.weight"));
        let out = shape[0];
        let inp = shape[1];
        Self {
            weight_b: block_weights(&weight, out, inp),
            out,
        }
    }
}

pub struct Linear {
    pub weight_b: Vec<f32>,
    pub bias: Vec<f32>,
    pub out: usize,
    pub inp: usize,
}

impl Linear {
    pub fn load(name: &str) -> Self {
        let (weight, shape) = load_weight(&format!("{name}.weight"));
        let out = shape[0];
        let inp = shape[1];
        let bias = load_weight(&format!("{name}.bias")).0;
        Self {
            weight_b: block_weights(&weight, out, inp),
            bias,
            out,
            inp,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.inp;
        let mut out = vec![0.0f32; rows * self.out];
        gemm(
            x,
            self.inp,
            rows,
            &self.weight_b,
            self.out,
            self.inp,
            Some(&self.bias),
            &mut out,
            self.out,
            false,
        );
        out
    }
}

pub fn block_weights(w: &[f32], n: usize, k: usize) -> Vec<f32> {
    let tiles = n.div_ceil(16);
    let mut out = vec![0.0f32; tiles * k * 16];
    block_rows_into(w, k, n, k, &mut out);
    out
}

pub fn block_rows_into(src: &[f32], stride: usize, n: usize, k: usize, dst: &mut [f32]) {
    let tiles = n.div_ceil(16);
    for jt in 0..tiles {
        let j0 = jt * 16;
        for t in 0..16 {
            let j = j0 + t;
            if j >= n {
                break;
            }
            let row = &src[j * stride..j * stride + k];
            for l in 0..k {
                dst[(jt * k + l) * 16 + t] = row[l];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &[f32],
    a_stride: usize,
    m: usize,
    b: &[f32],
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    out_stride: usize,
    accumulate: bool,
) {
    debug_assert!(m == 0 || a.len() >= (m - 1) * a_stride + k);
    debug_assert!(b.len() >= n.div_ceil(16) * k * 16);
    debug_assert!(m == 0 || out.len() >= (m - 1) * out_stride + n);

    gemm_core(
        a.as_ptr(),
        a_stride,
        m,
        b.as_ptr(),
        n,
        k,
        bias.map_or(core::ptr::null(), |b| b.as_ptr()),
        out.as_mut_ptr(),
        out_stride,
        accumulate,
    );
}

#[allow(clippy::too_many_arguments)]
fn gemm_core(
    a: *const f32,
    a_stride: usize,
    m: usize,
    b: *const f32,
    n: usize,
    k: usize,
    bias: *const f32,
    out: *mut f32,
    out_stride: usize,
    accumulate: bool,
) {
    unsafe {
        let has_bias = !bias.is_null();
        let tiles = n / 16;

        for jt in 0..tiles {
            let bj = b.add(jt * k * 16);
            let j = jt * 16;
            for i in 0..m {
                let mut acc = [0.0f32; 16];
                if has_bias {
                    for (t, a) in acc.iter_mut().enumerate() {
                        *a = *bias.add(j + t);
                    }
                }
                let mut ap = a.add(i * a_stride);
                let mut bp = bj;
                for _ in 0..k {
                    let x = *ap;
                    for (t, a) in acc.iter_mut().enumerate() {
                        *a = x.mul_add(*bp.add(t), *a);
                    }
                    ap = ap.add(1);
                    bp = bp.add(16);
                }
                let orow = out.add(i * out_stride + j);
                if accumulate {
                    for (t, a) in acc.iter().enumerate() {
                        *orow.add(t) = *orow.add(t) + a;
                    }
                } else {
                    for (t, a) in acc.iter().enumerate() {
                        *orow.add(t) = *a;
                    }
                }
            }
        }

        let done = tiles * 16;
        if done < n {
            for i in 0..m {
                let ar = a.add(i * a_stride);
                let orow = out.add(i * out_stride);
                for j in done..n {
                    let t = j - done;
                    let mut sum = if accumulate { *orow.add(j) } else { 0.0 };
                    if has_bias {
                        sum += *bias.add(j);
                    }
                    for l in 0..k {
                        sum = ar
                            .add(l)
                            .read()
                            .mul_add(*b.add((tiles * k + l) * 16 + t), sum);
                    }
                    *orow.add(j) = sum;
                }
            }
        }
    }
}

use crate::weights::load_weight;

pub fn quant_row(src: &[f32], dst: &mut [i8]) -> f32 {
    let n = src.len();
    let mut m4 = [0.0f32; 4];
    let mut c = 0;
    while c + 4 <= n {
        for t in 0..4 {
            let a = src[c + t].abs();
            if a > m4[t] {
                m4[t] = a;
            }
        }
        c += 4;
    }
    let mut mx = m4[0];
    for t in 1..4 {
        if m4[t] > mx {
            mx = m4[t];
        }
    }
    while c < n {
        let a = src[c].abs();
        if a > mx {
            mx = a;
        }
        c += 1;
    }
    if mx <= 0.0 {
        dst.fill(0);
        return 0.0;
    }
    let inv = 127.0 / mx;
    let scale = mx * (1.0 / 127.0);
    let mut c = 0;
    while c + 4 <= n {
        for t in 0..4 {
            dst[c + t] = (src[c + t] * inv) as i8;
        }
        c += 4;
    }
    while c < n {
        dst[c] = (src[c] * inv) as i8;
        c += 1;
    }
    scale
}

pub fn quant_rows(
    src: &[f32],
    src_stride: usize,
    dst: &mut [i8],
    dst_stride: usize,
    scales: &mut [f32],
    rows: usize,
    n: usize,
) {
    for r in 0..rows {
        let s = &src[r * src_stride..r * src_stride + n];
        scales[r] = quant_row(s, &mut dst[r * dst_stride..r * dst_stride + n]);
    }
}

const R: usize = 8;

#[inline(always)]
fn dot<const K: usize>(a: &[i8], b: &[i8]) -> i32 {
    let mut s = 0i32;
    for l in 0..K {
        s += (a[l] as i32) * (b[l] as i32);
    }
    s
}

#[inline(always)]
fn dot2<const K: usize>(a: &[i8], b0: &[i8], b1: &[i8]) -> (i32, i32) {
    let mut s0 = 0i32;
    let mut s1 = 0i32;
    for l in 0..K {
        let av = a[l] as i32;
        s0 += av * (b0[l] as i32);
        s1 += av * (b1[l] as i32);
    }
    (s0, s1)
}

#[inline(always)]
fn pair<const ACC: bool>(out: &mut [f32], j: usize, v0: f32, v1: f32) {
    let d = &mut out[j..j + 2];
    if ACC {
        d[0] += v0;
        d[1] += v1;
    } else {
        d[0] = v0;
        d[1] = v1;
    }
}

pub fn gemm<const K: usize, const ACC: bool>(
    a: &[i8],
    a_stride: usize,
    a_scale: &[f32],
    m: usize,
    w: &[i8],
    w_scale: &[f32],
    n: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    let nb = if K == 64 { 1 } else { 3 };
    gemm_nb::<K, ACC>(a, a_stride, a_scale, m, w, w_scale, n, out, out_stride, nb);
}

#[inline(never)]
fn gemm_nb<const K: usize, const ACC: bool>(
    a: &[i8],
    a_stride: usize,
    a_scale: &[f32],
    m: usize,
    w: &[i8],
    w_scale: &[f32],
    n: usize,
    out: &mut [f32],
    out_stride: usize,
    nb: usize,
) {
    let mut i = 0;
    while i + R <= m && (i + R) * out_stride <= out.len() && (i + R) <= a_scale.len() {
        let a0 = &a[i * a_stride..i * a_stride + K];
        let a1 = &a[(i + 1) * a_stride..(i + 1) * a_stride + K];
        let a2 = &a[(i + 2) * a_stride..(i + 2) * a_stride + K];
        let a3 = &a[(i + 3) * a_stride..(i + 3) * a_stride + K];
        let a4 = &a[(i + 4) * a_stride..(i + 4) * a_stride + K];
        let a5 = &a[(i + 5) * a_stride..(i + 5) * a_stride + K];
        let a6 = &a[(i + 6) * a_stride..(i + 6) * a_stride + K];
        let a7 = &a[(i + 7) * a_stride..(i + 7) * a_stride + K];
        let base = i * out_stride;
        let (r0, rest) = out[base..].split_at_mut(out_stride);
        let (r1, rest) = rest.split_at_mut(out_stride);
        let (r2, rest) = rest.split_at_mut(out_stride);
        let (r3, rest) = rest.split_at_mut(out_stride);
        let (r4, rest) = rest.split_at_mut(out_stride);
        let (r5, rest) = rest.split_at_mut(out_stride);
        let (r6, rest) = rest.split_at_mut(out_stride);
        let (r7, _) = rest.split_at_mut(out_stride);
        let s0 = a_scale[i];
        let s1 = a_scale[i + 1];
        let s2 = a_scale[i + 2];
        let s3 = a_scale[i + 3];
        let s4 = a_scale[i + 4];
        let s5 = a_scale[i + 5];
        let s6 = a_scale[i + 6];
        let s7 = a_scale[i + 7];
        let mut j = 0;
        if nb == 1 {
            while j < n {
                let b0 = &w[j * K..j * K + K];
                let w0 = w_scale[j];
                let v0 = s0 * w0 * dot::<K>(a0, b0) as f32;
                let v1 = s1 * w0 * dot::<K>(a1, b0) as f32;
                let v2 = s2 * w0 * dot::<K>(a2, b0) as f32;
                let v3 = s3 * w0 * dot::<K>(a3, b0) as f32;
                let v4 = s4 * w0 * dot::<K>(a4, b0) as f32;
                let v5 = s5 * w0 * dot::<K>(a5, b0) as f32;
                let v6 = s6 * w0 * dot::<K>(a6, b0) as f32;
                let v7 = s7 * w0 * dot::<K>(a7, b0) as f32;
                put::<ACC>(&mut r0[j..], v0);
                put::<ACC>(&mut r1[j..], v1);
                put::<ACC>(&mut r2[j..], v2);
                put::<ACC>(&mut r3[j..], v3);
                put::<ACC>(&mut r4[j..], v4);
                put::<ACC>(&mut r5[j..], v5);
                put::<ACC>(&mut r6[j..], v6);
                put::<ACC>(&mut r7[j..], v7);
                j += 1;
            }
        } else if nb == 3 {
            while j + 3 <= n {
                let b0 = &w[j * K..j * K + K];
                let b1 = &w[(j + 1) * K..(j + 1) * K + K];
                let b2 = &w[(j + 2) * K..(j + 2) * K + K];
                let w0 = w_scale[j];
                let w1 = w_scale[j + 1];
                let w2 = w_scale[j + 2];
                let (d0, d1, d2) = dot3::<K>(a0, b0, b1, b2);
                put3::<ACC>(&mut r0[j..], s0 * w0 * d0 as f32, s0 * w1 * d1 as f32, s0 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a1, b0, b1, b2);
                put3::<ACC>(&mut r1[j..], s1 * w0 * d0 as f32, s1 * w1 * d1 as f32, s1 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a2, b0, b1, b2);
                put3::<ACC>(&mut r2[j..], s2 * w0 * d0 as f32, s2 * w1 * d1 as f32, s2 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a3, b0, b1, b2);
                put3::<ACC>(&mut r3[j..], s3 * w0 * d0 as f32, s3 * w1 * d1 as f32, s3 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a4, b0, b1, b2);
                put3::<ACC>(&mut r4[j..], s4 * w0 * d0 as f32, s4 * w1 * d1 as f32, s4 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a5, b0, b1, b2);
                put3::<ACC>(&mut r5[j..], s5 * w0 * d0 as f32, s5 * w1 * d1 as f32, s5 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a6, b0, b1, b2);
                put3::<ACC>(&mut r6[j..], s6 * w0 * d0 as f32, s6 * w1 * d1 as f32, s6 * w2 * d2 as f32);
                let (d0, d1, d2) = dot3::<K>(a7, b0, b1, b2);
                put3::<ACC>(&mut r7[j..], s7 * w0 * d0 as f32, s7 * w1 * d1 as f32, s7 * w2 * d2 as f32);
                j += 3;
            }
            while j < n {
                let b0 = &w[j * K..j * K + K];
                let w0 = w_scale[j];
                let v0 = s0 * w0 * dot::<K>(a0, b0) as f32;
                let v1 = s1 * w0 * dot::<K>(a1, b0) as f32;
                let v2 = s2 * w0 * dot::<K>(a2, b0) as f32;
                let v3 = s3 * w0 * dot::<K>(a3, b0) as f32;
                let v4 = s4 * w0 * dot::<K>(a4, b0) as f32;
                let v5 = s5 * w0 * dot::<K>(a5, b0) as f32;
                let v6 = s6 * w0 * dot::<K>(a6, b0) as f32;
                let v7 = s7 * w0 * dot::<K>(a7, b0) as f32;
                put::<ACC>(&mut r0[j..], v0);
                put::<ACC>(&mut r1[j..], v1);
                put::<ACC>(&mut r2[j..], v2);
                put::<ACC>(&mut r3[j..], v3);
                put::<ACC>(&mut r4[j..], v4);
                put::<ACC>(&mut r5[j..], v5);
                put::<ACC>(&mut r6[j..], v6);
                put::<ACC>(&mut r7[j..], v7);
                j += 1;
            }
        } else {
            while j + 2 <= n {
                let b0 = &w[j * K..j * K + K];
                let b1 = &w[(j + 1) * K..(j + 1) * K + K];
                let w0 = w_scale[j];
                let w1 = w_scale[j + 1];
                let (d0, d1) = dot2::<K>(a0, b0, b1);
                pair::<ACC>(&mut r0[..], j, s0 * w0 * d0 as f32, s0 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a1, b0, b1);
                pair::<ACC>(&mut r1[..], j, s1 * w0 * d0 as f32, s1 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a2, b0, b1);
                pair::<ACC>(&mut r2[..], j, s2 * w0 * d0 as f32, s2 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a3, b0, b1);
                pair::<ACC>(&mut r3[..], j, s3 * w0 * d0 as f32, s3 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a4, b0, b1);
                pair::<ACC>(&mut r4[..], j, s4 * w0 * d0 as f32, s4 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a5, b0, b1);
                pair::<ACC>(&mut r5[..], j, s5 * w0 * d0 as f32, s5 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a6, b0, b1);
                pair::<ACC>(&mut r6[..], j, s6 * w0 * d0 as f32, s6 * w1 * d1 as f32);
                let (d0, d1) = dot2::<K>(a7, b0, b1);
                pair::<ACC>(&mut r7[..], j, s7 * w0 * d0 as f32, s7 * w1 * d1 as f32);
                j += 2;
            }
            while j < n {
                let b0 = &w[j * K..j * K + K];
                let w0 = w_scale[j];
                let v0 = s0 * w0 * dot::<K>(a0, b0) as f32;
                let v1 = s1 * w0 * dot::<K>(a1, b0) as f32;
                let v2 = s2 * w0 * dot::<K>(a2, b0) as f32;
                let v3 = s3 * w0 * dot::<K>(a3, b0) as f32;
                let v4 = s4 * w0 * dot::<K>(a4, b0) as f32;
                let v5 = s5 * w0 * dot::<K>(a5, b0) as f32;
                let v6 = s6 * w0 * dot::<K>(a6, b0) as f32;
                let v7 = s7 * w0 * dot::<K>(a7, b0) as f32;
                put::<ACC>(&mut r0[j..], v0);
                put::<ACC>(&mut r1[j..], v1);
                put::<ACC>(&mut r2[j..], v2);
                put::<ACC>(&mut r3[j..], v3);
                put::<ACC>(&mut r4[j..], v4);
                put::<ACC>(&mut r5[j..], v5);
                put::<ACC>(&mut r6[j..], v6);
                put::<ACC>(&mut r7[j..], v7);
                j += 1;
            }
        }
        i += R;
    }
    while i < m && (i + 1) * out_stride <= out.len() && (i + 1) <= a_scale.len() {
        let ar = &a[i * a_stride..i * a_stride + K];
        let sa = a_scale[i];
        let orow = &mut out[i * out_stride..i * out_stride + n];
        for j in 0..n {
            let br = &w[j * K..j * K + K];
            let v = sa * w_scale[j] * dot::<K>(ar, br) as f32;
            if ACC {
                orow[j] += v;
            } else {
                orow[j] = v;
            }
        }
        i += 1;
    }
}

#[inline(always)]
fn put<const ACC: bool>(o: &mut [f32], v: f32) {
    if ACC {
        o[0] += v;
    } else {
        o[0] = v;
    }
}

#[inline(always)]
fn put3<const ACC: bool>(o: &mut [f32], v0: f32, v1: f32, v2: f32) {
    let d = &mut o[..3];
    if ACC {
        d[0] += v0;
        d[1] += v1;
        d[2] += v2;
    } else {
        d[0] = v0;
        d[1] = v1;
        d[2] = v2;
    }
}

#[inline(always)]
fn dot3<const K: usize>(a: &[i8], b0: &[i8], b1: &[i8], b2: &[i8]) -> (i32, i32, i32) {
    let mut s0 = 0i32;
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    for l in 0..K {
        let av = a[l] as i32;
        s0 += av * (b0[l] as i32);
        s1 += av * (b1[l] as i32);
        s2 += av * (b2[l] as i32);
    }
    (s0, s1, s2)
}

fn gemm_dyn(
    a: &[i8],
    a_stride: usize,
    a_scale: &[f32],
    m: usize,
    w: &[i8],
    w_scale: &[f32],
    n: usize,
    k: usize,
    out: &mut [f32],
    out_stride: usize,
    acc: bool,
) {
    for i in 0..m {
        let ar = &a[i * a_stride..i * a_stride + k];
        let sa = a_scale[i];
        let orow = &mut out[i * out_stride..i * out_stride + n];
        for j in 0..n {
            let br = &w[j * k..j * k + k];
            let mut s = 0i32;
            for l in 0..k {
                s += (ar[l] as i32) * (br[l] as i32);
            }
            let v = sa * w_scale[j] * s as f32;
            if acc {
                orow[j] += v;
            } else {
                orow[j] = v;
            }
        }
    }
}

pub struct QMat {
    pub w: Vec<i8>,
    pub scale: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl QMat {
    pub fn from_f32(w: &[f32], n: usize, k: usize) -> Self {
        let mut q = vec![0i8; n * k];
        let mut scale = vec![0.0f32; n];
        quant_rows(w, k, &mut q, k, &mut scale, n, k);
        Self { w: q, scale, n, k }
    }

    pub fn load(name: &str) -> Self {
        let (w, shape) = load_weight(&format!("{name}.weight"));
        Self::from_f32(&w, shape[0], shape[1])
    }

    pub fn cat(a: &Self, b: &Self) -> Self {
        let mut w = Vec::with_capacity(a.w.len() + b.w.len());
        w.extend_from_slice(&a.w);
        w.extend_from_slice(&b.w);
        let mut scale = Vec::with_capacity(a.n + b.n);
        scale.extend_from_slice(&a.scale);
        scale.extend_from_slice(&b.scale);
        Self {
            w,
            scale,
            n: a.n + b.n,
            k: a.k,
        }
    }

    pub fn run(
        &self,
        a: &[i8],
        a_stride: usize,
        a_scale: &[f32],
        m: usize,
        out: &mut [f32],
        out_stride: usize,
        acc: bool,
    ) {
        match (self.k, acc) {
            (64, true) => gemm::<64, true>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            (64, false) => gemm::<64, false>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            (256, true) => gemm::<256, true>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            (256, false) => gemm::<256, false>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            (512, true) => gemm::<512, true>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            (512, false) => gemm::<512, false>(a, a_stride, a_scale, m, &self.w, &self.scale, self.n, out, out_stride),
            _ => gemm_dyn(
                a,
                a_stride,
                a_scale,
                m,
                &self.w,
                &self.scale,
                self.n,
                self.k,
                out,
                out_stride,
                acc,
            ),
        }
    }
}

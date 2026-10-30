use crate::ckpt::{Checkpoint, STResult};
use crate::global::{
    D, FFN, HALF, HEAD_DIM, KERNEL, MAX_LEN, N_ATTN, N_CONV, N_HEADS, NGRAM_BUCKETS, NGRAM_DIM,
    PAD, Q_SCALE,
};
use crate::math::{gelu, softmax_rows};
use crate::quant::{QMat, gemm, quant_row, quant_rows};
use fearless_simd::SimdBase;
use std::f32::consts::{LN_2, LOG2_E};
use std::fmt::Display;
use std::sync::Mutex;
#[cfg(not(feature = "wasm"))]
use std::time::Instant;
#[cfg(feature = "wasm")]
use web_time::Instant;

const LN_EPS: f32 = 1e-5;

#[inline]
fn silu_one(v: f32) -> f32 {
    let d = -v * LOG2_E;
    let nn = (d + 12582912.0) - 12582912.0;
    let rr = d - nn;
    let mut pp = 0.05550411;
    pp = rr.mul_add(pp, 0.240_226_5);
    pp = rr.mul_add(pp, LN_2);
    pp = rr.mul_add(pp, 1.0);
    let ni = (nn as i32).clamp(-126, 127);
    let ex = pp * f32::from_bits(((ni + 127) as u32) << 23);
    let den = 1.0 + ex;
    let r0 = f32::from_bits(0x7EF311C3 - den.to_bits());
    let r1 = r0 * (2.0 - den * r0);
    v * r1
}

fn row_max<S: fearless_simd::Simd>(simd: S, row: &[f32]) -> f32 {
    use fearless_simd::f32x4;
    let mut mv = simd.splat_f32x4(f32::NEG_INFINITY);
    let mut c = 0;
    while c + 4 <= row.len() {
        mv = simd.max_f32x4(mv, f32x4::from_slice(simd, &row[c..c + 4]));
        c += 4;
    }
    let a = mv.to_array();
    let mut mx = a[0];
    for t in a.iter().skip(1) {
        if *t > mx {
            mx = *t;
        }
    }
    while c < row.len() {
        if row[c] > mx {
            mx = row[c];
        }
        c += 1;
    }
    mx
}

fn softmax_tile_s<S: fearless_simd::Simd>(
    simd: S,
    stile: &mut [f32],
    sq: &mut [i8],
    ss: &mut [f32],
    rlen: usize,
    rb: usize,
) {
    for r in 0..rlen {
        let g = rb + r;
        let row = &stile[r * MAX_LEN..r * MAX_LEN + MAX_LEN];
        let dst = &mut sq[g * MAX_LEN..g * MAX_LEN + MAX_LEN];
        let mx = row_max(simd, row);
        let mut s4 = [0.0f32; 4];
        let mut c = 0;
        while c + 4 <= MAX_LEN {
            for t in 0..4 {
                let v = row[c + t];
                let d = (v - mx) * LOG2_E;
                let nn = (d + 12582912.0) - 12582912.0;
                let rr = d - nn;
                let mut pp = 0.240_226_5;
                pp = rr.mul_add(pp, LN_2);
                pp = rr.mul_add(pp, 1.0);
                let ni = (nn as i32).clamp(-126, 127);
                let e = pp * f32::from_bits(((ni + 127) as u32) << 23);
                dst[c + t] = (e * 127.0 + 0.5) as i8;
                s4[t] += e;
            }
            c += 4;
        }
        let sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
        ss[g] = 1.0 / (127.0 * sum);
    }
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(ckpt: &Checkpoint, prefix: String) -> STResult<Self> {
        Ok(Self {
            weight: ckpt.f32(format!("{prefix}.weight"))?.0,
            bias: ckpt.f32(format!("{prefix}.bias"))?.0,
        })
    }

    fn ln(&self, src: &[f32], dst: &mut [f32], start: usize, end: usize) {
        if start >= end {
            return;
        }
        fearless_simd::dispatch!(
            fearless_simd::Level::new(),
            simd => ln_rows(simd, &self.weight, &self.bias, src, dst, start, end)
        );
    }
}

fn ln_rows<S: fearless_simd::Simd>(
    simd: S,
    w: &[f32],
    b: &[f32],
    src: &[f32],
    dst: &mut [f32],
    start: usize,
    end: usize,
) {
    use fearless_simd::{SimdBase, f32x4};
    let zero = simd.splat_f32x4(0.0);
    for r in start..end {
        let base = r * D;
        let row = &src[base..base + D];
        let out = &mut dst[base..base + D];
        let mut acc = zero;
        let mut c = 0;
        while c + 4 <= D {
            acc = simd.add_f32x4(acc, f32x4::from_slice(simd, &row[c..c + 4]));
            c += 4;
        }
        let a = acc.to_array();
        let mean = ((a[0] + a[1]) + (a[2] + a[3])) / D as f32;
        let mv = simd.splat_f32x4(mean);
        let mut acc = zero;
        let mut c = 0;
        while c + 4 <= D {
            let d = simd.sub_f32x4(f32x4::from_slice(simd, &row[c..c + 4]), mv);
            acc = simd.mul_add_f32x4(d, d, acc);
            c += 4;
        }
        let v = acc.to_array();
        let var = ((v[0] + v[1]) + (v[2] + v[3])) / D as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        let iv = simd.splat_f32x4(inv);
        let mut i = 0;
        while i + 4 <= D {
            let rv = simd.sub_f32x4(f32x4::from_slice(simd, &row[i..i + 4]), mv);
            let wv = f32x4::from_slice(simd, &w[i..i + 4]);
            let bv = f32x4::from_slice(simd, &b[i..i + 4]);
            let o = simd.mul_add_f32x4(rv, simd.mul_f32x4(iv, wv), bv);
            out[i..i + 4].copy_from_slice(&o.to_array());
            i += 4;
        }
    }
}

struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn load() -> Self {
        let mut cos = vec![0.0f32; MAX_LEN * HALF];
        let mut sin = vec![0.0f32; MAX_LEN * HALF];
        let mut inv_freq = [0.0f32; HALF];
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            *freq = 1.0f32 / 10000.0f32.powf((2 * i) as f32 / HEAD_DIM as f32);
        }
        for pos in 0..MAX_LEN {
            for i in 0..HALF {
                let angle = pos as f32 * inv_freq[i];
                cos[pos * HALF + i] = angle.cos();
                sin[pos * HALF + i] = angle.sin();
            }
        }
        Self { cos, sin }
    }

    fn forward(&self, qkv: &mut [f32]) {
        for pos in 0..MAX_LEN {
            let cbase = pos * HALF;
            let row = pos * 3 * D;
            for head in 0..N_HEADS {
                let qb = row + head * HEAD_DIM;
                let kb = qb + D;
                for i in 0..HALF {
                    let c = self.cos[cbase + i];
                    let s = self.sin[cbase + i];

                    let a = qkv[qb + i];
                    let b = qkv[qb + i + HALF];
                    qkv[qb + i] = (a * c - b * s) * Q_SCALE;
                    qkv[qb + i + HALF] = (b * c + a * s) * Q_SCALE;

                    let a = qkv[kb + i];
                    let b = qkv[kb + i + HALF];
                    qkv[kb + i] = a * c - b * s;
                    qkv[kb + i + HALF] = b * c + a * s;
                }
            }
        }
    }
}

struct Ffn {
    upgate: QMat,
    down: QMat,
}

impl Ffn {
    fn load(ckpt: &Checkpoint, prefix: String) -> STResult<Self> {
        Ok(Self {
            upgate: QMat::cat(
                &QMat::load(ckpt, format!("{prefix}.ffn_gate"))?,
                &QMat::load(ckpt, format!("{prefix}.ffn_up"))?,
            ),
            down: QMat::load(ckpt, format!("{prefix}.ffn_down"))?,
        })
    }
}

struct ConvLayer {
    norm1: Norm,
    conv_weight: Vec<f32>,
    conv_bias: Vec<f32>,
    norm2: Norm,
    ffn: Ffn,
}

impl ConvLayer {
    fn load(ckpt: &Checkpoint, prefix: String) -> STResult<Self> {
        let (weight, shape) = ckpt.f32(format!("{prefix}.conv.weight"))?;
        let out = shape[0];
        let kernel = *shape.last().unwrap();
        let mut conv_weight = vec![0.0f32; out * kernel];
        for o in 0..out {
            for k in 0..kernel {
                conv_weight[k * out + o] = weight[o * kernel + k];
            }
        }
        Ok(Self {
            norm1: Norm::load(ckpt, format!("{prefix}.norm1"))?,
            conv_weight,
            conv_bias: ckpt.f32(format!("{prefix}.conv.bias"))?.0,
            norm2: Norm::load(ckpt, format!("{prefix}.norm2"))?,
            ffn: Ffn::load(ckpt, prefix)?,
        })
    }
}

struct AttnLayer {
    norm1: Norm,
    qkv: QMat,
    out_proj: QMat,
    norm2: Norm,
    ffn: Ffn,
}

impl AttnLayer {
    fn load(ckpt: &Checkpoint, prefix: String) -> STResult<Self> {
        Ok(Self {
            norm1: Norm::load(ckpt, format!("{prefix}.norm1"))?,
            qkv: QMat::load(ckpt, format!("{prefix}.qkv"))?,
            out_proj: QMat::load(ckpt, format!("{prefix}.out_proj"))?,
            norm2: Norm::load(ckpt, format!("{prefix}.norm2"))?,
            ffn: Ffn::load(ckpt, prefix)?,
        })
    }
}

pub struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    hq: Vec<i8>,
    hs: Vec<f32>,
    conv: Vec<f32>,
    buf: Vec<f32>,
    bufq: Vec<i8>,
    bufs: Vec<f32>,
    ngram: Vec<f32>,
    nq: Vec<i8>,
    ns: Vec<f32>,
    qkv: Vec<f32>,
    qq: Vec<i8>,
    qs: Vec<f32>,
    kq: Vec<i8>,
    ks: Vec<f32>,
    stile: Vec<f32>,
    sq: Vec<i8>,
    ss: Vec<f32>,
    vb: Vec<i8>,
    vbs: Vec<f32>,
    attn: Vec<f32>,
    aq: Vec<i8>,
    asc: Vec<f32>,
    pooled: Vec<f32>,
    hid: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self {
            x: vec![0.0f32; MAX_LEN * D],
            h: vec![0.0f32; MAX_LEN * D],
            hq: vec![0i8; MAX_LEN * D],
            hs: vec![0.0f32; MAX_LEN],
            conv: vec![0.0f32; MAX_LEN * D],
            buf: vec![0.0f32; MAX_LEN * 2 * FFN],
            bufq: vec![0i8; MAX_LEN * FFN],
            bufs: vec![0.0f32; MAX_LEN],
            ngram: vec![0.0f32; MAX_LEN * NGRAM_DIM],
            nq: vec![0i8; MAX_LEN * NGRAM_DIM],
            ns: vec![0.0f32; MAX_LEN],
            qkv: vec![0.0f32; MAX_LEN * 3 * D],
            qq: vec![0i8; MAX_LEN * HEAD_DIM],
            qs: vec![0.0f32; MAX_LEN],
            kq: vec![0i8; MAX_LEN * HEAD_DIM],
            ks: vec![0.0f32; MAX_LEN],
            stile: vec![0.0f32; 64 * MAX_LEN],
            sq: vec![0i8; MAX_LEN * MAX_LEN],
            ss: vec![0.0f32; MAX_LEN],
            vb: vec![0i8; HEAD_DIM * MAX_LEN],
            vbs: vec![0.0f32; HEAD_DIM],
            attn: vec![0.0f32; MAX_LEN * D],
            aq: vec![0i8; MAX_LEN * D],
            asc: vec![0.0f32; MAX_LEN],
            pooled: vec![0.0f32; D],
            hid: vec![0.0f32; D],
        }
    }

    fn norm_q(&mut self, norm: &Norm, start: usize, end: usize) {
        if start >= end {
            return;
        }
        norm.ln(&self.x, &mut self.h, start, end);
        quant_rows(
            &self.h[start * D..end * D],
            D,
            &mut self.hq[start * D..end * D],
            D,
            &mut self.hs[start..end],
            end - start,
            D,
        );
    }

    fn ffn<const MODE: u8>(&mut self, ffn: &Ffn, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let len = end - start;
        let n = ffn.upgate.n;
        ffn.upgate.run::<MODE>(
            &self.hq[start * D..],
            D,
            &self.hs[start..],
            len,
            &mut self.buf[start * n..],
            n,
            false,
        );
        let half = n / 2;
        for r in 0..len {
            let base = (start + r) * n;
            let (gate, rest) = self.buf[base..base + n].split_at_mut(half);
            for (g, u) in gate.iter_mut().zip(rest[..half].iter()) {
                let x = *g;
                let d = -x * LOG2_E;
                let nn = (d + 12582912.0) - 12582912.0;
                let rr = d - nn;
                let mut pp = 0.05550411;
                pp = rr.mul_add(pp, 0.240_226_5);
                pp = rr.mul_add(pp, LN_2);
                pp = rr.mul_add(pp, 1.0);
                let ni = (nn as i32).clamp(-126, 127);
                let ex = pp * f32::from_bits(((ni + 127) as u32) << 23);
                *g = (*u) * x / (1.0 + ex);
            }
            let gq = &mut self.bufq[(start + r) * half..(start + r) * half + half];
            self.bufs[start + r] = quant_row(&self.buf[base..base + half], gq);
        }
        ffn.down.run::<MODE>(
            &self.bufq[start * half..],
            half,
            &self.bufs[start..],
            len,
            &mut self.x[start * D..],
            D,
            true,
        );
    }

    fn conv_rows(&mut self, weight: &[f32], bias: &[f32], start: usize, end: usize) {
        for pos in start..end {
            let first = if pos + 1 >= KERNEL {
                0
            } else {
                KERNEL - 1 - pos
            };
            let mut ob = 0;
            while ob + 32 <= D {
                let mut acc = [0.0f32; 32];
                acc.copy_from_slice(&bias[ob..ob + 32]);
                for k in first..KERNEL {
                    let src = pos + k - (KERNEL - 1);
                    let hr = &self.h[src * D + ob..src * D + ob + 32];
                    let wr = &weight[k * D + ob..k * D + ob + 32];
                    for t in 0..32 {
                        acc[t] = wr[t].mul_add(hr[t], acc[t]);
                    }
                }
                for acc_t in &mut acc {
                    *acc_t = silu_one(*acc_t);
                }
                let row = &mut self.conv[pos * D + ob..pos * D + ob + 32];
                row.copy_from_slice(&acc);
                ob += 32;
            }
        }
    }
    fn conv_forward<const MODE: u8>(&mut self, layer: &ConvLayer, stable: usize) -> usize {
        let r = if stable >= 510 { 510 } else { stable + 14 };
        if r + 1 >= 496 {
            layer.norm1.ln(&self.x, &mut self.h, 0, MAX_LEN);
        } else {
            layer.norm1.ln(&self.x, &mut self.h, 0, r + 1);
            layer.norm1.ln(&self.x, &mut self.h, 496, MAX_LEN);
        }

        if r >= 510 {
            self.conv_rows(&layer.conv_weight, &layer.conv_bias, 0, MAX_LEN);
        } else {
            self.conv_rows(&layer.conv_weight, &layer.conv_bias, 0, r + 1);
            self.conv_rows(&layer.conv_weight, &layer.conv_bias, 510, MAX_LEN);
            for pos in r + 1..510 {
                self.conv.copy_within(r * D..r * D + D, pos * D);
            }
        }

        let head_end = if r >= 510 { MAX_LEN } else { r + 1 };
        for (xv, cv) in self.x[..head_end * D]
            .iter_mut()
            .zip(self.conv[..head_end * D].iter())
        {
            *xv += *cv;
        }
        if r < 510 {
            for (xv, cv) in self.x[510 * D..]
                .iter_mut()
                .zip(self.conv[510 * D..].iter())
            {
                *xv += *cv;
            }
        }

        if r >= 510 {
            self.norm_q(&layer.norm2, 0, MAX_LEN);
            self.ffn::<MODE>(&layer.ffn, 0, MAX_LEN);
            return 510;
        }
        self.norm_q(&layer.norm2, 0, r + 1);
        self.norm_q(&layer.norm2, 510, MAX_LEN);
        self.ffn::<MODE>(&layer.ffn, 0, r + 1);
        self.ffn::<MODE>(&layer.ffn, 510, MAX_LEN);
        for pos in r + 1..510 {
            self.x.copy_within(r * D..r * D + D, pos * D);
        }
        r
    }

    fn softmax_tile(&mut self, rlen: usize, rb: usize) {
        if rlen == 0 {
            return;
        }
        fearless_simd::dispatch!(
            fearless_simd::Level::new(),
            simd => softmax_tile_s(simd, &mut self.stile, &mut self.sq, &mut self.ss, rlen, rb)
        );
    }

    fn quant_v(&mut self, voff: usize) {
        let mut mx = [0.0f32; HEAD_DIM];
        let mut pos = 0;
        while pos + 4 <= MAX_LEN {
            for t in 0..4 {
                let src = &self.qkv[(pos + t) * 3 * D + voff..(pos + t) * 3 * D + voff + HEAD_DIM];
                for (j, v) in src.iter().enumerate() {
                    let a = v.abs();
                    if a > mx[j] {
                        mx[j] = a;
                    }
                }
            }
            pos += 4;
        }
        while pos < MAX_LEN {
            let src = &self.qkv[pos * 3 * D + voff..pos * 3 * D + voff + HEAD_DIM];
            for (j, v) in src.iter().enumerate() {
                let a = v.abs();
                if a > mx[j] {
                    mx[j] = a;
                }
            }
            pos += 1;
        }
        let mut inv = [0.0f32; HEAD_DIM];
        for j in 0..HEAD_DIM {
            if mx[j] > 0.0 {
                inv[j] = 127.0 / mx[j];
            }
            self.vbs[j] = mx[j] * (1.0 / 127.0);
        }
        for pos in 0..MAX_LEN {
            let src = &self.qkv[pos * 3 * D + voff..pos * 3 * D + voff + HEAD_DIM];
            for j in 0..HEAD_DIM {
                self.vb[j * MAX_LEN + pos] = (src[j] * inv[j]).round() as i8;
            }
        }
    }
    fn attn_forward<const MODE: u8>(
        &mut self,
        layer: &AttnLayer,
        rope: &Rope,
        stable: usize,
        rows: usize,
    ) {
        let cut = if stable >= 509 { 510 } else { stable + 1 };
        if cut >= 510 {
            layer.norm1.ln(&self.x, &mut self.h, 0, MAX_LEN);
        } else {
            layer.norm1.ln(&self.x, &mut self.h, 0, cut);
            layer.norm1.ln(&self.x, &mut self.h, 510, MAX_LEN);
        }
        if cut >= 510 {
            quant_rows(&self.h, D, &mut self.hq, D, &mut self.hs, MAX_LEN, D);
            layer
                .qkv
                .run::<MODE>(&self.hq, D, &self.hs, MAX_LEN, &mut self.qkv, 3 * D, false);
        } else {
            quant_rows(
                &self.h[0..cut * D],
                D,
                &mut self.hq[0..cut * D],
                D,
                &mut self.hs[0..cut],
                cut,
                D,
            );
            layer.qkv.run::<MODE>(
                &self.hq[0..cut * D],
                D,
                &self.hs[0..cut],
                cut,
                &mut self.qkv,
                3 * D,
                false,
            );
            quant_rows(
                &self.h[510 * D..],
                D,
                &mut self.hq[510 * D..],
                D,
                &mut self.hs[510..],
                2,
                D,
            );
            layer.qkv.run::<MODE>(
                &self.hq[510 * D..],
                D,
                &self.hs[510..],
                2,
                &mut self.qkv[510 * 3 * D..],
                3 * D,
                false,
            );
            for pos in cut..510 {
                self.qkv
                    .copy_within(stable * 3 * D..stable * 3 * D + 3 * D, pos * 3 * D);
            }
        }
        rope.forward(&mut self.qkv);

        for head in 0..N_HEADS {
            let qoff = head * HEAD_DIM;
            let koff = D + head * HEAD_DIM;
            let voff = 2 * D + head * HEAD_DIM;

            for pos in 0..rows {
                let src = &self.qkv[pos * 3 * D + qoff..pos * 3 * D + qoff + HEAD_DIM];
                self.qs[pos] =
                    quant_row(src, &mut self.qq[pos * HEAD_DIM..pos * HEAD_DIM + HEAD_DIM]);
            }
            for pos in 0..MAX_LEN {
                let src = &self.qkv[pos * 3 * D + koff..pos * 3 * D + koff + HEAD_DIM];
                self.ks[pos] =
                    quant_row(src, &mut self.kq[pos * HEAD_DIM..pos * HEAD_DIM + HEAD_DIM]);
            }

            let mut rb = 0;
            while rb < rows {
                let rend = if rb + 64 < rows { rb + 64 } else { rows };
                let rlen = rend - rb;
                gemm::<HEAD_DIM, false, MODE>(
                    &self.qq[rb * HEAD_DIM..],
                    HEAD_DIM,
                    &self.qs[rb..],
                    rlen,
                    &self.kq,
                    &self.ks,
                    MAX_LEN,
                    &mut self.stile,
                    MAX_LEN,
                );
                self.softmax_tile(rlen, rb);
                rb = rend;
            }
            self.quant_v(voff);
            gemm::<MAX_LEN, false, MODE>(
                &self.sq,
                MAX_LEN,
                &self.ss,
                rows,
                &self.vb,
                &self.vbs,
                HEAD_DIM,
                &mut self.attn[qoff..],
                D,
            );
        }

        quant_rows(&self.attn, D, &mut self.aq, D, &mut self.asc, rows, D);
        layer
            .out_proj
            .run::<MODE>(&self.aq, D, &self.asc, rows, &mut self.x, D, true);

        self.norm_q(&layer.norm2, 0, rows);
        self.ffn::<MODE>(&layer.ffn, 0, rows);
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Profile {
    pub mode: u8,
}

impl Default for Profile {
    fn default() -> Self {
        Self { mode: 2 }
    }
}

impl Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "simd_mode: {}",
            match self.mode {
                2 => "manual",
                1 => "auto_wide",
                _ => "auto",
            }
        )
    }
}

pub struct Model {
    embed: Vec<f32>,
    ngram_embed: Vec<f32>,
    ngram_proj: QMat,
    conv_layers: Vec<ConvLayer>,
    attn_layers: Vec<AttnLayer>,
    rope: Rope,
    final_norm: Norm,
    head0w: Vec<f32>,
    head0_bias: Vec<f32>,
    head3w: Vec<f32>,
    head3_bias: Vec<f32>,
    pub profile: Profile,
    scratch: Mutex<Scratch>,
}

impl Model {
    pub fn load(b: &[u8]) -> STResult<Self> {
        let ckpt = Checkpoint::from_bytes(b)?;
        let mut conv_layers = Vec::with_capacity(N_CONV);
        for i in 0..N_CONV {
            conv_layers.push(ConvLayer::load(&ckpt, format!("conv_layers.{i}"))?);
        }
        let mut attn_layers = Vec::with_capacity(N_ATTN);
        for i in 0..N_ATTN {
            attn_layers.push(AttnLayer::load(&ckpt, format!("attn_layers.{i}"))?);
        }
        Ok(Self {
            embed: ckpt.f32("embed.weight".to_string())?.0,
            ngram_embed: ckpt.f32("ngram_embed.embed.weight".to_string())?.0,
            ngram_proj: QMat::load(&ckpt, "ngram_proj".to_string())?,
            conv_layers,
            attn_layers,
            rope: Rope::load(),
            final_norm: Norm::load(&ckpt, "final_norm".to_string())?,
            head0w: ckpt.f32("head.0.weight".to_string())?.0,
            head0_bias: ckpt.f32("head.0.bias".to_string())?.0,
            head3w: ckpt.f32("head.3.weight".to_string())?.0,
            head3_bias: ckpt.f32("head.3.bias".to_string())?.0,
            profile: Profile::default(),
            scratch: Mutex::new(Scratch::new()),
        })
    }
    pub fn tune(&mut self) {
        let q = &self.ngram_proj;
        let m = 12;
        let k = q.k;
        let n = q.n;
        let a = vec![1i8; m * k];
        let sc = vec![1.0f32; m];
        let mut out = vec![0.0f32; m * n];
        q.run::<0>(&a, k, &sc, m, &mut out, n, false);
        q.run::<1>(&a, k, &sc, m, &mut out, n, false);
        q.run::<2>(&a, k, &sc, m, &mut out, n, false);
        let mut best = 0u8;
        let mut best_t = u128::MAX;
        for mode in 0..3u8 {
            let now = Instant::now();
            for _ in 0..8 {
                match mode {
                    0 => q.run::<0>(&a, k, &sc, m, &mut out, n, false),
                    1 => q.run::<1>(&a, k, &sc, m, &mut out, n, false),
                    _ => q.run::<2>(&a, k, &sc, m, &mut out, n, false),
                }
            }
            let t = now.elapsed().as_micros();
            if t < best_t {
                best_t = t;
                best = mode;
            }
        }
        self.profile.mode = best;
    }

    fn ngram_into(&self, byte_ids: &[usize], start: usize, end: usize, dst: &mut [f32]) {
        for pos in start..end {
            let id = byte_ids[pos];
            let b0 = id.min(255);
            let b1 = if pos + 1 < MAX_LEN {
                byte_ids[pos + 1].min(255)
            } else {
                0
            };
            let b2 = if pos + 2 < MAX_LEN {
                byte_ids[pos + 2].min(255)
            } else {
                0
            };
            let bucket = (b0 * 257 * 257 + b1 * 257 + b2) % NGRAM_BUCKETS;
            dst[pos * NGRAM_DIM..pos * NGRAM_DIM + NGRAM_DIM].copy_from_slice(
                &self.ngram_embed[bucket * NGRAM_DIM..bucket * NGRAM_DIM + NGRAM_DIM],
            );
        }
    }

    pub fn predict_with<const MODE: u8>(&self, byte_ids: &[usize], s: &mut Scratch) -> Vec<f32> {
        if byte_ids.len() != MAX_LEN {
            let mut buf = vec![PAD; MAX_LEN];
            for (dst, &src) in buf.iter_mut().zip(byte_ids.iter().take(MAX_LEN)) {
                *dst = src;
            }
            return self.predict_with::<MODE>(&buf, s);
        }
        let rows = byte_ids
            .iter()
            .rposition(|&id| id != PAD)
            .map_or(0, |p| p + 1);
        if rows == 0 {
            s.pooled.fill(0.0);
            return self.head(s);
        }

        let full = rows >= 510;
        let stop = if full { MAX_LEN } else { rows + 1 };

        for (pos, id) in byte_ids.iter().enumerate().take(stop) {
            let e = (*id).min(PAD);
            s.x[pos * D..pos * D + D].copy_from_slice(&self.embed[e * D..e * D + D]);
        }
        if !full {
            for (pos, id) in byte_ids.iter().enumerate().take(MAX_LEN).skip(510) {
                let e = (*id).min(PAD);
                s.x[pos * D..pos * D + D].copy_from_slice(&self.embed[e * D..e * D + D]);
            }
        }
        self.ngram_into(byte_ids, 0, stop, &mut s.ngram);
        if !full {
            self.ngram_into(byte_ids, 510, MAX_LEN, &mut s.ngram);
        }
        quant_rows(
            &s.ngram, NGRAM_DIM, &mut s.nq, NGRAM_DIM, &mut s.ns, stop, NGRAM_DIM,
        );
        self.ngram_proj
            .run::<MODE>(&s.nq, NGRAM_DIM, &s.ns, stop, &mut s.x, D, true);
        if !full {
            quant_rows(
                &s.ngram[510 * NGRAM_DIM..],
                NGRAM_DIM,
                &mut s.nq[510 * NGRAM_DIM..],
                NGRAM_DIM,
                &mut s.ns[510..],
                2,
                NGRAM_DIM,
            );
            self.ngram_proj.run::<MODE>(
                &s.nq[510 * NGRAM_DIM..],
                NGRAM_DIM,
                &s.ns[510..],
                2,
                &mut s.x[510 * D..],
                D,
                true,
            );
            for pos in stop..510 {
                s.x.copy_within(rows * D..rows * D + D, pos * D);
            }
        }

        let mut stable = rows.min(510);
        for layer in &self.conv_layers {
            stable = s.conv_forward::<MODE>(layer, stable);
        }
        for layer in &self.attn_layers {
            s.attn_forward::<MODE>(layer, &self.rope, stable, rows);
        }
        self.final_norm.ln(&s.x, &mut s.h, 0, rows);

        let mut count = 0usize;
        s.pooled.fill(0.0);
        for (pos, byte_id) in byte_ids.iter().enumerate().take(MAX_LEN) {
            if *byte_id != PAD {
                count += 1;
                let row = &s.h[pos * D..pos * D + D];
                for (p, value) in s.pooled.iter_mut().zip(row.iter()) {
                    *p += *value;
                }
            }
        }
        let denom = count.max(1) as f32;
        for value in s.pooled.iter_mut() {
            *value /= denom;
        }
        self.head(s)
    }

    pub fn predict(&self, byte_ids: &[usize]) -> Vec<f32> {
        let mut buf;
        let ids = if byte_ids.len() == MAX_LEN {
            byte_ids
        } else {
            buf = vec![PAD; MAX_LEN];
            for (dst, &src) in buf.iter_mut().zip(byte_ids.iter().take(MAX_LEN)) {
                *dst = src;
            }
            buf.as_slice()
        };
        let mut scr = self.scratch.lock().unwrap();
        match self.profile.mode {
            2 => self.predict_with::<2>(ids, &mut scr),
            1 => self.predict_with::<1>(ids, &mut scr),
            _ => self.predict_with::<0>(ids, &mut scr),
        }
    }

    fn head(&self, s: &mut Scratch) -> Vec<f32> {
        let n3 = self.head3_bias.len();
        for o in 0..D {
            let mut acc = self.head0_bias[o];
            for l in 0..D {
                acc = self.head0w[o * D + l].mul_add(s.pooled[l], acc);
            }
            s.hid[o] = gelu(acc);
        }
        let mut logits = vec![0.0f32; n3];
        for (j, logit) in logits.iter_mut().enumerate().take(n3) {
            let mut acc = self.head3_bias[j];
            let row = &self.head3w[j * D..j * D + D];
            for (l, r) in row.iter().enumerate().take(D) {
                acc = r.mul_add(s.hid[l], acc);
            }
            *logit = acc;
        }
        softmax_rows(&mut logits, n3);
        logits
    }
}

use crate::fp32::math::{gelu, silu_inplace, softmax_rows};

const LN_EPS: f32 = 1e-5;
use crate::global::{
    D, FFN, HALF, HEAD_DIM, KERNEL, MAX_LEN, N_ATTN, N_CONV, N_HEADS, NGRAM_BUCKETS, NGRAM_DIM,
    PAD, Q_SCALE,
};
use crate::int8::quant::{gemm, quant_row, quant_rows, QMat};
use crate::weights::load_weight;

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn load(name: &str) -> Self {
        Self {
            weight: load_weight(&format!("{name}.weight")).0,
            bias: load_weight(&format!("{name}.bias")).0,
        }
    }

    fn ln(&self, src: &[f32], dst: &mut [f32], start: usize, end: usize) {
        if start >= end {
            return;
        }
        for r in start..end {
            let base = r * D;
            let row = &src[base..base + D];
            let out = &mut dst[base..base + D];
            let mut a4 = [0.0f32; 4];
            let mut c = 0;
            while c + 4 <= D {
                for t in 0..4 {
                    a4[t] += row[c + t];
                }
                c += 4;
            }
            let mean = ((a4[0] + a4[1]) + (a4[2] + a4[3])) / D as f32;
            let mut v4 = [0.0f32; 4];
            let mut c = 0;
            while c + 4 <= D {
                for t in 0..4 {
                    let diff = row[c + t] - mean;
                    v4[t] = diff.mul_add(diff, v4[t]);
                }
                c += 4;
            }
            let var = ((v4[0] + v4[1]) + (v4[2] + v4[3])) / D as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for i in 0..D {
                out[i] = (row[i] - mean).mul_add(inv * self.weight[i], self.bias[i]);
            }
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
    fn load(prefix: &str) -> Self {
        Self {
            upgate: QMat::cat(
                &QMat::load(&format!("{prefix}.ffn_gate")),
                &QMat::load(&format!("{prefix}.ffn_up")),
            ),
            down: QMat::load(&format!("{prefix}.ffn_down")),
        }
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
    fn load(prefix: &str) -> Self {
        let (weight, shape) = load_weight(&format!("{prefix}.conv.weight"));
        let out = shape[0];
        let kernel = *shape.last().unwrap();
        let mut conv_weight = vec![0.0f32; out * kernel];
        for o in 0..out {
            for k in 0..kernel {
                conv_weight[k * out + o] = weight[o * kernel + k];
            }
        }
        Self {
            norm1: Norm::load(&format!("{prefix}.norm1")),
            conv_weight,
            conv_bias: load_weight(&format!("{prefix}.conv.bias")).0,
            norm2: Norm::load(&format!("{prefix}.norm2")),
            ffn: Ffn::load(prefix),
        }
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
    fn load(prefix: &str) -> Self {
        Self {
            norm1: Norm::load(&format!("{prefix}.norm1")),
            qkv: QMat::load(&format!("{prefix}.qkv")),
            out_proj: QMat::load(&format!("{prefix}.out_proj")),
            norm2: Norm::load(&format!("{prefix}.norm2")),
            ffn: Ffn::load(prefix),
        }
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

    fn ffn(&mut self, ffn: &Ffn, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let len = end - start;
        let n = ffn.upgate.n;
        ffn.upgate.run(
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
                let d = -x * 1.442695;
                let nn = (d + 12582912.0) - 12582912.0;
                let rr = d - nn;
                let mut pp = 0.009618129;
                pp = rr.mul_add(pp, 0.055504109);
                pp = rr.mul_add(pp, 0.24022651);
                pp = rr.mul_add(pp, 0.69314718);
                pp = rr.mul_add(pp, 1.0);
                let ni = (nn as i32).clamp(-126, 127);
                let ex = pp * f32::from_bits(((ni + 127) as u32) << 23);
                *g = (*u) * x / (1.0 + ex);
            }
        }
        quant_rows(
            &self.buf[start * n..start * n + len * n],
            n,
            &mut self.bufq[start * half..start * half + len * half],
            half,
            &mut self.bufs[start..end],
            len,
            half,
        );
        ffn.down.run(
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
                silu_inplace(&mut acc);
                let row = &mut self.conv[pos * D + ob..pos * D + ob + 32];
                row.copy_from_slice(&acc);
                ob += 32;
            }
        }
    }
    fn conv_forward(&mut self, layer: &ConvLayer, stable: usize) -> usize {
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
            self.ffn(&layer.ffn, 0, MAX_LEN);
            return 510;
        }
        self.norm_q(&layer.norm2, 0, r + 1);
        self.norm_q(&layer.norm2, 510, MAX_LEN);
        self.ffn(&layer.ffn, 0, r + 1);
        self.ffn(&layer.ffn, 510, MAX_LEN);
        for pos in r + 1..510 {
            self.x.copy_within(r * D..r * D + D, pos * D);
        }
        r
    }

    fn softmax_tile(&mut self, rlen: usize, rb: usize) {
        for r in 0..rlen {
            let row = &mut self.stile[r * MAX_LEN..r * MAX_LEN + MAX_LEN];
            let mut m4 = [f32::NEG_INFINITY; 4];
            let mut c = 0;
            while c + 4 <= MAX_LEN {
                for t in 0..4 {
                    let v = row[c + t];
                    if v > m4[t] {
                        m4[t] = v;
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
            let mut s4 = [0.0f32; 4];
            let mut c = 0;
            while c + 4 <= MAX_LEN {
                for t in 0..4 {
                    let v = row[c + t];
                    let d = (v - mx) * 1.442695;
                    let nn = (d + 12582912.0) - 12582912.0;
                    let rr = d - nn;
                    let mut pp = 0.009618129;
                    pp = rr.mul_add(pp, 0.055504109);
                    pp = rr.mul_add(pp, 0.24022651);
                    pp = rr.mul_add(pp, 0.69314718);
                    pp = rr.mul_add(pp, 1.0);
                    let ni = (nn as i32).clamp(-126, 127);
                    let e = pp * f32::from_bits(((ni + 127) as u32) << 23);
                    row[c + t] = e;
                    s4[t] += e;
                }
                c += 4;
            }
            let sum = (s4[0] + s4[1]) + (s4[2] + s4[3]);
            let scale = 127.0 / sum;
            let g = rb + r;
            let dst = &mut self.sq[g * MAX_LEN..g * MAX_LEN + MAX_LEN];
            for (d, v) in dst.iter_mut().zip(row.iter()) {
                *d = (*v * scale) as i8;
            }
            self.ss[g] = sum * (1.0 / 127.0);
        }
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
                self.vb[j * MAX_LEN + pos] = (src[j] * inv[j]) as i8;
            }
        }
    }

    fn attn_forward(&mut self, layer: &AttnLayer, rope: &Rope, stable: usize, rows: usize) {
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
                .run(&self.hq, D, &self.hs, MAX_LEN, &mut self.qkv, 3 * D, false);
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
            layer.qkv.run(
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
            layer.qkv.run(
                &self.hq[510 * D..],
                D,
                &self.hs[510..],
                2,
                &mut self.qkv[510 * 3 * D..],
                3 * D,
                false,
            );
            for pos in cut..510 {
                self.qkv.copy_within(stable * 3 * D..stable * 3 * D + 3 * D, pos * 3 * D);
            }
        }
        rope.forward(&mut self.qkv);

        for head in 0..N_HEADS {
            let qoff = head * HEAD_DIM;
            let koff = D + head * HEAD_DIM;
            let voff = 2 * D + head * HEAD_DIM;

            for pos in 0..rows {
                let src = &self.qkv[pos * 3 * D + qoff..pos * 3 * D + qoff + HEAD_DIM];
                self.qs[pos] = quant_row(src, &mut self.qq[pos * HEAD_DIM..pos * HEAD_DIM + HEAD_DIM]);
            }
            for pos in 0..MAX_LEN {
                let src = &self.qkv[pos * 3 * D + koff..pos * 3 * D + koff + HEAD_DIM];
                self.ks[pos] = quant_row(src, &mut self.kq[pos * HEAD_DIM..pos * HEAD_DIM + HEAD_DIM]);
            }

            let mut rb = 0;
            while rb < rows {
                let rend = if rb + 64 < rows { rb + 64 } else { rows };
                let rlen = rend - rb;
                gemm::<HEAD_DIM, false>(
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
            gemm::<MAX_LEN, false>(
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
            .run(&self.aq, D, &self.asc, rows, &mut self.x, D, true);

        self.norm_q(&layer.norm2, 0, rows);
        self.ffn(&layer.ffn, 0, rows);
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
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
}

impl Model {
    pub fn load() -> Self {
        let mut conv_layers = Vec::with_capacity(N_CONV);
        for i in 0..N_CONV {
            conv_layers.push(ConvLayer::load(&format!("conv_layers.{i}")));
        }
        let mut attn_layers = Vec::with_capacity(N_ATTN);
        for i in 0..N_ATTN {
            attn_layers.push(AttnLayer::load(&format!("attn_layers.{i}")));
        }
        Self {
            embed: load_weight("embed.weight").0,
            ngram_embed: load_weight("ngram_embed.embed.weight").0,
            ngram_proj: QMat::load("ngram_proj"),
            conv_layers,
            attn_layers,
            rope: Rope::load(),
            final_norm: Norm::load("final_norm"),
            head0w: load_weight("head.0.weight").0,
            head0_bias: load_weight("head.0.bias").0,
            head3w: load_weight("head.3.weight").0,
            head3_bias: load_weight("head.3.bias").0,
        }
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
            dst[pos * NGRAM_DIM..pos * NGRAM_DIM + NGRAM_DIM]
                .copy_from_slice(&self.ngram_embed[bucket * NGRAM_DIM..bucket * NGRAM_DIM + NGRAM_DIM]);
        }
    }

    pub fn predict_with(&self, byte_ids: &[usize], s: &mut Scratch) -> Vec<f32> {
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

        for pos in 0..stop {
            let id = byte_ids[pos];
            s.x[pos * D..pos * D + D].copy_from_slice(&self.embed[id * D..id * D + D]);
        }
        if !full {
            for pos in 510..MAX_LEN {
                let id = byte_ids[pos];
                s.x[pos * D..pos * D + D].copy_from_slice(&self.embed[id * D..id * D + D]);
            }
        }
        self.ngram_into(byte_ids, 0, stop, &mut s.ngram);
        if !full {
            self.ngram_into(byte_ids, 510, MAX_LEN, &mut s.ngram);
        }

        quant_rows(
            &s.ngram,
            NGRAM_DIM,
            &mut s.nq,
            NGRAM_DIM,
            &mut s.ns,
            stop,
            NGRAM_DIM,
        );
        self.ngram_proj
            .run(&s.nq, NGRAM_DIM, &s.ns, stop, &mut s.x, D, true);
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
            self.ngram_proj.run(
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
            stable = s.conv_forward(layer, stable);
        }
        for layer in &self.attn_layers {
            s.attn_forward(layer, &self.rope, stable, rows);
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
        let mut s = Scratch::new();
        self.predict_with(byte_ids, &mut s)
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
        for j in 0..n3 {
            let mut acc = self.head3_bias[j];
            let row = &self.head3w[j * D..j * D + D];
            for l in 0..D {
                acc = row[l].mul_add(s.hid[l], acc);
            }
            logits[j] = acc;
        }
        softmax_rows(&mut logits, n3);
        logits
    }
}

use std::sync::Mutex;

mod linear;
mod math;
use linear::{Linear, NoBiasLinear, block_rows_into, gemm};
use math::{gelu, layer_norm_into, silu_inplace, softmax, softmax_rows, swiglu_inplace};
mod weights;
use weights::load_weight;

const D: usize = 256;
const N_CONV: usize = 3;
const N_ATTN: usize = 1;
const N_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HALF: usize = HEAD_DIM / 2;
const KERNEL: usize = 15;
const NGRAM_BUCKETS: usize = 4096;
const NGRAM_DIM: usize = 64;
pub static MAX_LEN: usize = 512;
const NUM_CLASSES: usize = 334;
const FFN: usize = 512;
const PAD: usize = 256;
const Q_SCALE: f32 = 0.125;

pub static LANGS: [&str; NUM_CLASSES] = [
    "abk", "ace", "ach", "ady", "afr", "aka", "akk", "alt", "amh", "ami", "ang", "anp", "arb",
    "arc", "arg", "arq", "ary", "arz", "asm", "ast", "atj", "ava", "avk", "awa", "aym", "azb",
    "azj", "bak", "bam", "ban", "bar", "bbc", "bel", "ben", "bho", "bik", "bis", "bjn", "blk",
    "bod", "bos", "bpy", "bre", "bug", "bul", "bxr", "cat", "cbk", "cdo", "ceb", "ces", "che",
    "chr", "chu", "chv", "chy", "ckb", "cor", "cos", "crh", "csb", "cym", "dag", "dan", "deu",
    "din", "diq", "div", "dsb", "dty", "dzo", "egx", "egy", "ell", "eml", "eng", "epo", "est",
    "ett", "eus", "ewe", "ext", "fao", "fas", "fij", "fin", "fon", "fra", "fro", "frp", "frr",
    "fry", "ful", "fur", "gag", "gan", "gaz", "gcr", "gez", "gla", "gle", "glg", "glk", "glv",
    "gom", "gor", "got", "gpe", "grc", "gsw", "guc", "gug", "guj", "gur", "guw", "hak", "hat",
    "hau", "haw", "hbo", "hbs", "heb", "hif", "hin", "hrv", "hsb", "hun", "hye", "hyw", "ibo",
    "ido", "iku", "ile", "ilo", "ina", "ind", "inh", "ipk", "isl", "ita", "jam", "jav", "jbo",
    "jpn", "kaa", "kab", "kal", "kan", "kas", "kat", "kaz", "kbd", "kbp", "kcg", "khk", "khm",
    "kik", "kin", "kir", "kmr", "koi", "kom", "kon", "kor", "krc", "ksh", "lad", "lao", "lat",
    "latex", "lav", "lbe", "lez", "lfn", "lij", "lim", "lin", "lit", "lld", "lmo", "ltg", "ltz",
    "lug", "luo", "lzh", "mad", "mai", "mal", "mar", "mdf", "mhr", "min", "mkd", "mlg", "mlt",
    "mni", "mnw", "mri", "mrj", "msa", "mwl", "mya", "myv", "mzn", "nah", "nan", "nap", "nav",
    "nds", "nep", "new", "nia", "nld", "nno", "nor", "nov", "nqo", "nrf", "nso", "nya", "nyn",
    "oci", "olo", "orm", "ory", "oss", "pag", "pam", "pan", "pap", "pcd", "pcm", "pdc", "peo",
    "pfl", "pms", "pnb", "pnt", "pol", "por", "pus", "pwn", "quy", "rcf", "rmy", "roh", "ron",
    "rue", "run", "rup", "rus", "sah", "san", "sat", "scn", "sgs", "shi", "shn", "sin", "skr",
    "slk", "slv", "sme", "smn", "smo", "sna", "snd", "som", "sot", "spa", "sqi", "srd", "srn",
    "srp", "ssw", "stq", "sun", "sux", "swe", "swh", "szl", "szy", "tah", "tam", "tat", "tay",
    "tcy", "tel", "tet", "tgk", "tgl", "tha", "tir", "tly", "ton", "tpi", "trv", "tsn", "tso",
    "tuk", "tum", "tur", "txb", "tyv", "udm", "uig", "ukr", "urd", "uzb", "vec", "ven", "vep",
    "vie", "vls", "vol", "vro", "war", "wln", "wol", "wuu", "xal", "xcl", "xho", "xmf", "xog",
    "xto", "ydd", "yor", "yue", "zea", "zgh", "zha", "zho", "zul",
];

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

    fn forward(&self, src: &[f32], dst: &mut [f32], rows: usize) {
        layer_norm_into(src, dst, rows, D, &self.weight, &self.bias);
    }

    fn forward_rows(&self, src: &[f32], dst: &mut [f32], ranges: &[(usize, usize)]) {
        for &(start, end) in ranges {
            if start < end {
                layer_norm_into(
                    &src[start * D..],
                    &mut dst[start * D..],
                    end - start,
                    D,
                    &self.weight,
                    &self.bias,
                );
            }
        }
    }
}

struct Ffn {
    upgate: NoBiasLinear,
    down: NoBiasLinear,
}

impl Ffn {
    fn load(prefix: &str) -> Self {
        let (gate, shape) = load_weight(&format!("{prefix}.ffn_gate.weight"));
        let (up, _) = load_weight(&format!("{prefix}.ffn_up.weight"));
        let n = shape[0];
        let k = shape[1];
        let stride = 2 * n;
        let tiles = stride.div_ceil(16);
        let mut weight_b = vec![0.0f32; tiles * k * 16];
        for jt in 0..tiles {
            let j0 = jt * 16;
            for t in 0..16 {
                let j = j0 + t;
                if j >= stride {
                    break;
                }
                let src = if j < n {
                    &gate[j * k..j * k + k]
                } else {
                    &up[(j - n) * k..(j - n) * k + k]
                };
                for l in 0..k {
                    weight_b[(jt * k + l) * 16 + t] = src[l];
                }
            }
        }
        Self {
            upgate: NoBiasLinear {
                weight_b,
                out: stride,
            },
            down: NoBiasLinear::load(&format!("{prefix}.ffn_down")),
        }
    }

    fn forward(&self, x: &mut [f32], h: &[f32], buf: &mut [f32], rows: usize) {
        self.forward_ranges(x, h, buf, &[(0, rows)]);
    }

    fn forward_ranges(&self, x: &mut [f32], h: &[f32], buf: &mut [f32], ranges: &[(usize, usize)]) {
        let n = self.upgate.out;
        for &(start, end) in ranges {
            let len = end - start;
            if len == 0 {
                continue;
            }
            gemm(
                &h[start * D..],
                D,
                len,
                &self.upgate.weight_b,
                n,
                D,
                None,
                &mut buf[start * n..],
                n,
                false,
            );
            swiglu_inplace(&mut buf[start * n..], len, n / 2);
            gemm(
                &buf[start * n..],
                n,
                len,
                &self.down.weight_b,
                D,
                FFN,
                None,
                &mut x[start * D..],
                D,
                true,
            );
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

    fn conv_rows(&self, h: &[f32], conv: &mut [f32], start: usize, end: usize) {
        for pos in start..end {
            let row = &mut conv[pos * D..pos * D + D];
            row.copy_from_slice(&self.conv_bias);
            let first = if pos + 1 >= KERNEL {
                0
            } else {
                KERNEL - 1 - pos
            };
            for k in first..KERNEL {
                let src = pos + k - (KERNEL - 1);
                let hr = &h[src * D..src * D + D];
                let wr = &self.conv_weight[k * D..k * D + D];
                for o in 0..D {
                    row[o] = wr[o].mul_add(hr[o], row[o]);
                }
            }
            silu_inplace(row);
        }
    }

    fn forward(
        &self,
        x: &mut [f32],
        h: &mut [f32],
        conv: &mut [f32],
        buf: &mut [f32],
        stable: usize,
    ) -> usize {
        let r = if stable >= 510 { 510 } else { stable + 14 };
        if r + 1 >= 496 {
            self.norm1.forward(x, h, MAX_LEN);
        } else {
            self.norm1.forward_rows(x, h, &[(0, r + 1), (496, MAX_LEN)]);
        }

        let mut row = [0.0f32; D];
        if r >= 510 {
            self.conv_rows(h, conv, 0, MAX_LEN);
        } else {
            self.conv_rows(h, conv, 0, r + 1);
            self.conv_rows(h, conv, 510, MAX_LEN);
            row.copy_from_slice(&conv[r * D..r * D + D]);
            for pos in r + 1..510 {
                conv[pos * D..pos * D + D].copy_from_slice(&row);
            }
        }

        for (xv, cv) in x.iter_mut().zip(conv.iter()) {
            *xv += *cv;
        }

        if r >= 510 {
            self.norm2.forward(x, h, MAX_LEN);
            self.ffn.forward(x, h, buf, MAX_LEN);
            return 510;
        }
        self.norm2.forward_rows(x, h, &[(0, r + 1), (510, MAX_LEN)]);
        self.ffn
            .forward_ranges(x, h, buf, &[(0, r + 1), (510, MAX_LEN)]);
        row.copy_from_slice(&x[r * D..r * D + D]);
        for pos in r + 1..510 {
            x[pos * D..pos * D + D].copy_from_slice(&row);
        }
        r
    }
}

struct AttnLayer {
    norm1: Norm,
    qkv: NoBiasLinear,
    out_proj: NoBiasLinear,
    norm2: Norm,
    ffn: Ffn,
}

impl AttnLayer {
    fn load(prefix: &str) -> Self {
        Self {
            norm1: Norm::load(&format!("{prefix}.norm1")),
            qkv: NoBiasLinear::load(&format!("{prefix}.qkv")),
            out_proj: NoBiasLinear::load(&format!("{prefix}.out_proj")),
            norm2: Norm::load(&format!("{prefix}.norm2")),
            ffn: Ffn::load(prefix),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &mut [f32],
        h: &mut [f32],
        qkv: &mut [f32],
        attn: &mut [f32],
        scores: &mut [f32],
        kt: &mut [f32],
        vb: &mut [f32],
        buf: &mut [f32],
        rope: &Rope,
        stable: usize,
        rows: usize,
    ) {
        let cut = if stable >= 509 { 510 } else { stable + 1 };
        if cut >= 510 {
            self.norm1.forward(x, h, MAX_LEN);
        } else {
            self.norm1.forward_rows(x, h, &[(0, cut), (510, MAX_LEN)]);
        }
        if cut >= 510 {
            gemm(
                h,
                D,
                MAX_LEN,
                &self.qkv.weight_b,
                3 * D,
                D,
                None,
                qkv,
                3 * D,
                false,
            );
        } else {
            gemm(
                h,
                D,
                cut,
                &self.qkv.weight_b,
                3 * D,
                D,
                None,
                qkv,
                3 * D,
                false,
            );
            gemm(
                &h[510 * D..],
                D,
                2,
                &self.qkv.weight_b,
                3 * D,
                D,
                None,
                &mut qkv[510 * 3 * D..],
                3 * D,
                false,
            );
            let mut row = [0.0f32; 3 * D];
            row.copy_from_slice(&qkv[stable * 3 * D..stable * 3 * D + 3 * D]);
            for pos in cut..510 {
                qkv[pos * 3 * D..pos * 3 * D + 3 * D].copy_from_slice(&row);
            }
        }
        rope.forward(qkv);

        for head in 0..N_HEADS {
            let qoff = head * HEAD_DIM;
            let koff = D + head * HEAD_DIM;
            let voff = 2 * D + head * HEAD_DIM;

            block_rows_into(&qkv[koff..], 3 * D, MAX_LEN, HEAD_DIM, kt);

            gemm(
                &qkv[qoff..],
                3 * D,
                rows,
                kt,
                MAX_LEN,
                HEAD_DIM,
                None,
                scores,
                MAX_LEN,
                false,
            );
            softmax_rows(&mut scores[..rows * MAX_LEN], MAX_LEN);

            for jt in 0..HEAD_DIM / 16 {
                let j0 = jt * 16;
                for s in 0..MAX_LEN {
                    let src = &qkv[s * 3 * D + voff + j0..s * 3 * D + voff + j0 + 16];
                    vb[(jt * MAX_LEN + s) * 16..(jt * MAX_LEN + s) * 16 + 16].copy_from_slice(src);
                }
            }

            gemm(
                scores,
                MAX_LEN,
                rows,
                vb,
                HEAD_DIM,
                MAX_LEN,
                None,
                &mut attn[qoff..],
                D,
                false,
            );
        }

        gemm(
            attn,
            D,
            rows,
            &self.out_proj.weight_b,
            D,
            D,
            None,
            x,
            D,
            true,
        );

        self.norm2.forward(x, h, rows);
        self.ffn.forward(x, h, buf, rows);
    }
}

struct Scratch {
    x: Vec<f32>,
    h: Vec<f32>,
    conv: Vec<f32>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    scores: Vec<f32>,
    kt: Vec<f32>,
    vb: Vec<f32>,
    buf: Vec<f32>,
    ngram: Vec<f32>,
    pooled: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            x: vec![0.0f32; MAX_LEN * D],
            h: vec![0.0f32; MAX_LEN * D],
            conv: vec![0.0f32; MAX_LEN * D],
            qkv: vec![0.0f32; MAX_LEN * 3 * D],
            attn: vec![0.0f32; MAX_LEN * D],
            scores: vec![0.0f32; MAX_LEN * MAX_LEN],
            kt: vec![0.0f32; HEAD_DIM * MAX_LEN],
            vb: vec![0.0f32; HEAD_DIM * MAX_LEN],
            buf: vec![0.0f32; MAX_LEN * 2 * FFN],
            ngram: vec![0.0f32; MAX_LEN * NGRAM_DIM],
            pooled: vec![0.0f32; D],
        }
    }
}

pub struct Model {
    embed: Vec<f32>,
    ngram_embed: Vec<f32>,
    ngram_proj: NoBiasLinear,
    conv_layers: Vec<ConvLayer>,
    attn_layers: Vec<AttnLayer>,
    rope: Rope,
    final_norm: Norm,
    head0: Linear,
    head3: Linear,
    scratch: Mutex<Scratch>,
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
            ngram_proj: NoBiasLinear::load("ngram_proj"),
            conv_layers,
            attn_layers,
            rope: Rope::load(),
            final_norm: Norm::load("final_norm"),
            head0: Linear::load("head.0"),
            head3: Linear::load("head.3"),
            scratch: Mutex::new(Scratch::new()),
        }
    }

    pub fn predict(&self, byte_ids: &[usize]) -> Vec<f32> {
        let mut scratch = self.scratch.lock().unwrap();
        self.predict_with(byte_ids, &mut scratch)
    }

    fn predict_with(&self, byte_ids: &[usize], s: &mut Scratch) -> Vec<f32> {
        let rows = byte_ids
            .iter()
            .rposition(|&id| id != PAD)
            .map_or(0, |p| p + 1);
        if rows == 0 {
            s.pooled.fill(0.0);
            let hidden = self.head0.forward(&s.pooled);
            let activated: Vec<f32> = hidden.into_iter().map(gelu).collect();
            let mut logits = self.head3.forward(&activated);
            softmax(&mut logits);
            return logits;
        }

        let x = &mut s.x;
        let full = rows >= 510;
        let stop = if full { MAX_LEN } else { rows + 1 };
        for pos in 0..stop {
            let id = byte_ids[pos];
            x[pos * D..pos * D + D].copy_from_slice(&self.embed[id * D..id * D + D]);

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
            s.ngram[pos * NGRAM_DIM..pos * NGRAM_DIM + NGRAM_DIM]
                .copy_from_slice(&self.ngram_embed[bucket * NGRAM_DIM..][..NGRAM_DIM]);
        }
        if full {
            gemm(
                &s.ngram,
                NGRAM_DIM,
                MAX_LEN,
                &self.ngram_proj.weight_b,
                D,
                NGRAM_DIM,
                None,
                x,
                D,
                true,
            );
        } else {
            gemm(
                &s.ngram,
                NGRAM_DIM,
                stop,
                &self.ngram_proj.weight_b,
                D,
                NGRAM_DIM,
                None,
                x,
                D,
                true,
            );
            let mut row = [0.0f32; D];
            row.copy_from_slice(&x[rows * D..rows * D + D]);
            for pos in stop..510 {
                x[pos * D..pos * D + D].copy_from_slice(&row);
            }
            for pos in 510..MAX_LEN {
                let id = byte_ids[pos];
                x[pos * D..pos * D + D].copy_from_slice(&self.embed[id * D..id * D + D]);
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
                s.ngram[pos * NGRAM_DIM..pos * NGRAM_DIM + NGRAM_DIM]
                    .copy_from_slice(&self.ngram_embed[bucket * NGRAM_DIM..][..NGRAM_DIM]);
            }
            gemm(
                &s.ngram[510 * NGRAM_DIM..],
                NGRAM_DIM,
                2,
                &self.ngram_proj.weight_b,
                D,
                NGRAM_DIM,
                None,
                &mut x[510 * D..],
                D,
                true,
            );
        }
        let mut stable = rows.min(510);
        for layer in &self.conv_layers {
            stable = layer.forward(&mut s.x, &mut s.h, &mut s.conv, &mut s.buf, stable);
        }
        for layer in &self.attn_layers {
            layer.forward(
                &mut s.x,
                &mut s.h,
                &mut s.qkv,
                &mut s.attn,
                &mut s.scores,
                &mut s.kt,
                &mut s.vb,
                &mut s.buf,
                &self.rope,
                stable,
                rows,
            );
        }
        self.final_norm.forward(&s.x, &mut s.h, rows);

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

        let hidden = self.head0.forward(&s.pooled);
        let activated: Vec<f32> = hidden.into_iter().map(gelu).collect();
        let mut logits = self.head3.forward(&activated);
        softmax(&mut logits);
        logits
    }
}

pub fn encode(text: &str) -> Vec<usize> {
    let mut ids = vec![PAD; MAX_LEN];
    let raw = text.as_bytes();
    let len = raw.len().min(MAX_LEN);
    for i in 0..len {
        ids[i] = raw[i] as usize;
    }
    ids
}

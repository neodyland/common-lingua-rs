use itertools::multizip;

static D: usize = 256;
static N_CONV: usize = 3;
static N_ATTN: usize = 1;
static N_HEADS: usize = 4;
static HEAD_DIM: usize = 64;
static KERNEL: usize = 15;
static NGRAM_BUCKETS: usize = 4096;
static NGRAM_DIM: usize = 64;
pub static MAX_LEN: usize = 512;
static NUM_CLASSES: usize = 334;
static FFN: usize = 512;
static PAD: usize = 256;
static NORM_EPS: f32 = 1e-5;
mod linear;
mod math;
use linear::{Linear, NoBiasLinear};
use math::{gelu, silu, softmax};
mod weights;
use weights::load_weight;

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
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn load() -> Self {
        let half = HEAD_DIM / 2;
        let mut cos = vec![0.0f32; MAX_LEN * half];
        let mut sin = vec![0.0f32; MAX_LEN * half];
        for pos in 0..MAX_LEN {
            for i in 0..half {
                let freq = 1.0f32 / 10000.0f32.powf((2 * i) as f32 / HEAD_DIM as f32);
                let angle = pos as f32 * freq;
                cos[pos * half + i] = angle.cos();
                sin[pos * half + i] = angle.sin();
            }
        }
        Self { sin, cos, half }
    }
    fn forward(&self, q: &mut [f32], k: &mut [f32]) {
        for head in 0..N_HEADS {
            let base = head * MAX_LEN * HEAD_DIM;
            for pos in 0..MAX_LEN {
                for i in 0..self.half {
                    let index = base + pos * HEAD_DIM + i;
                    let c = self.cos[pos * self.half + i];
                    let s = self.sin[pos * self.half + i];
                    let a = q[index];
                    let b = q[index + self.half];
                    q[index] = a * c - b * s;
                    q[index + self.half] = b * c + a * s;
                    let a = k[index];
                    let b = k[index + self.half];
                    k[index] = a * c - b * s;
                    k[index + self.half] = b * c + a * s;
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
    fn forward(&self, x: &mut [f32]) {
        for row in x.as_chunks_mut::<D>().0 {
            let mut mean = 0.0f32;
            for value in row.iter() {
                mean += value;
            }
            mean /= D as f32;
            let mut var = 0.0f32;
            for value in row.iter() {
                let diff = value - mean;
                var += diff * diff;
            }
            var /= D as f32;
            let inv = 1.0f32 / (var + NORM_EPS).sqrt();
            for i in 0..D {
                row[i] = (row[i] - mean) * inv * self.weight[i] + self.bias[i];
            }
        }
    }
}

struct Ffn {
    gate: NoBiasLinear,
    up: NoBiasLinear,
    down: NoBiasLinear,
}
#[inline(always)]
fn hsum8(a: [f32; 8]) -> f32 {
    (a[0] + a[1]) + (a[2] + a[3]) + (a[4] + a[5]) + (a[6] + a[7])
}

static FFN_BATCH: usize = 8;
const _: () = assert!(D % FFN_BATCH == 0);
static FFN_PB: usize = 4;
const _: () = assert!(MAX_LEN % FFN_PB == 0);

impl Ffn {
    fn load(prefix: &str) -> Self {
        Self {
            gate: NoBiasLinear::load(&format!("{prefix}.ffn_gate")),
            up: NoBiasLinear::load(&format!("{prefix}.ffn_up")),
            down: NoBiasLinear::load(&format!("{prefix}.ffn_down")),
        }
    }
    fn forward(&self, x: &mut [f32], h: &[f32]) {
        /*let mut gate = vec![0.0f32; FFN];
        for pos in 0..MAX_LEN {
            for j in 0..FFN {
                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;
                let gate_row = &self.gate.weight[j * D..j * D + D];
                let up_row = &self.up.weight[j * D..j * D + D];
                for i in 0..D {
                    let value = h[pos * D + i];
                    gate_sum += gate_row[i] * value;
                    up_sum += up_row[i] * value;
                }
                gate[j] = silu(gate_sum) * up_sum;
            }
            for o in 0..D {
                let mut sum = if self.down.bias.is_empty() {
                    0.0f32
                } else {
                    self.down.bias[o]
                };
                let row = &self.down.weight[o * FFN..o * FFN + FFN];
                for j in 0..FFN {
                    sum += row[j] * gate[j];
                }
                x[pos * D + o] += sum;
            }
        }*/
        let mut gate_buf = vec![0.0f32; FFN_PB * FFN];

        let gate_w = &self.gate.weight[..];
        let up_w = &self.up.weight[..];
        let down_w = &self.down.weight[..];

        for (hb, xb) in (h.chunks_exact(FFN_PB * D)).zip(&mut x.chunks_exact_mut(FFN_PB * D)) {
            for (j, (gr, ur)) in gate_w.chunks_exact(D).zip(up_w.chunks_exact(D)).enumerate() {
                let mut sg0 = [0.0f32; FFN_BATCH];
                let mut sg1 = [0.0f32; FFN_BATCH];
                let mut sg2 = [0.0f32; FFN_BATCH];
                let mut sg3 = [0.0f32; FFN_BATCH];
                let mut su0 = [0.0f32; FFN_BATCH];
                let mut su1 = [0.0f32; FFN_BATCH];
                let mut su2 = [0.0f32; FFN_BATCH];
                let mut su3 = [0.0f32; FFN_BATCH];

                for (g, u, v0, v1, v2, v3) in multizip((
                    gr.chunks_exact(FFN_BATCH),
                    ur.chunks_exact(FFN_BATCH),
                    hb[0..D].chunks_exact(FFN_BATCH),
                    hb[D..D * 2].chunks_exact(FFN_BATCH),
                    hb[D * 2..D * 3].chunks_exact(FFN_BATCH),
                    hb[D * 3..D * 4].chunks_exact(FFN_BATCH),
                )) {
                    for k in 0..FFN_BATCH {
                        let g = g[k];
                        let u = u[k];
                        let (v0, v1, v2, v3) = (v0[k], v1[k], v2[k], v3[k]);
                        sg0[k] += g * v0;
                        sg1[k] += g * v1;
                        sg2[k] += g * v2;
                        sg3[k] += g * v3;
                        su0[k] += u * v0;
                        su1[k] += u * v1;
                        su2[k] += u * v2;
                        su3[k] += u * v3;
                    }
                }
                gate_buf[j] = silu(hsum8(sg0)) * hsum8(su0);
                gate_buf[FFN + j] = silu(hsum8(sg1)) * hsum8(su1);
                gate_buf[2 * FFN + j] = silu(hsum8(sg2)) * hsum8(su2);
                gate_buf[3 * FFN + j] = silu(hsum8(sg3)) * hsum8(su3);
            }

            let g0 = &gate_buf[..FFN];
            let g1 = &gate_buf[FFN..2 * FFN];
            let g2 = &gate_buf[2 * FFN..3 * FFN];
            let g3 = &gate_buf[3 * FFN..4 * FFN];
            for (o, wr) in down_w.chunks_exact(FFN).enumerate() {
                let mut s0 = [0.0f32; FFN_BATCH];
                let mut s1 = [0.0f32; FFN_BATCH];
                let mut s2 = [0.0f32; FFN_BATCH];
                let mut s3 = [0.0f32; FFN_BATCH];
                for (wr_0, g0_0, g1_0, g2_0, g3_0) in multizip((
                    wr.chunks_exact(FFN_BATCH),
                    g0.chunks_exact(FFN_BATCH),
                    g1.chunks_exact(FFN_BATCH),
                    g2.chunks_exact(FFN_BATCH),
                    g3.chunks_exact(FFN_BATCH),
                )) {
                    for k in 0..FFN_BATCH {
                        let w = wr_0[k];
                        s0[k] += w * g0_0[k];
                        s1[k] += w * g1_0[k];
                        s2[k] += w * g2_0[k];
                        s3[k] += w * g3_0[k];
                    }
                }
                xb[o] += hsum8(s0);
                xb[D + o] += hsum8(s1);
                xb[2 * D + o] += hsum8(s2);
                xb[3 * D + o] += hsum8(s3);
            }
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
        Self {
            norm1: Norm::load(&format!("{prefix}.norm1")),
            conv_weight: load_weight(&format!("{prefix}.conv.weight")).0,
            conv_bias: load_weight(&format!("{prefix}.conv.bias")).0,
            norm2: Norm::load(&format!("{prefix}.norm2")),
            ffn: Ffn::load(prefix),
        }
    }
    fn forward(&self, x: &mut Vec<f32>) {
        let mut h = x.clone();
        self.norm1.forward(&mut h);
        let mut conv = vec![0.0f32; MAX_LEN * D];
        for pos in 0..MAX_LEN {
            for o in 0..D {
                let mut sum = self.conv_bias[o];
                let row = &self.conv_weight[o * KERNEL..o * KERNEL + KERNEL];
                for k in 0..KERNEL {
                    let src = pos as isize + k as isize - (KERNEL - 1) as isize;
                    if src >= 0 {
                        sum += row[k] * h[src as usize * D + o];
                    }
                }
                conv[pos * D + o] = silu(sum);
            }
        }
        for i in 0..x.len() {
            x[i] += conv[i];
        }
        h = x.clone();
        self.norm2.forward(&mut h);
        self.ffn.forward(x, &h);
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
    fn forward(&self, x: &mut Vec<f32>, rope: &Rope) {
        let mut h = x.clone();
        self.norm1.forward(&mut h);
        let qkv = self.qkv.forward(&h);
        let mut q = vec![0.0f32; N_HEADS * MAX_LEN * HEAD_DIM];
        let mut k = vec![0.0f32; N_HEADS * MAX_LEN * HEAD_DIM];
        let mut v = vec![0.0f32; N_HEADS * MAX_LEN * HEAD_DIM];
        for pos in 0..MAX_LEN {
            for head in 0..N_HEADS {
                let dst = head * MAX_LEN * HEAD_DIM + pos * HEAD_DIM;
                let src = pos * 3 * D + head * HEAD_DIM;
                q[dst..dst + HEAD_DIM].copy_from_slice(&qkv[src..src + HEAD_DIM]);
                k[dst..dst + HEAD_DIM].copy_from_slice(&qkv[src + D..src + D + HEAD_DIM]);
                v[dst..dst + HEAD_DIM].copy_from_slice(&qkv[src + 2 * D..src + 2 * D + HEAD_DIM]);
            }
        }
        rope.forward(&mut q, &mut k);
        let mut attn_out = vec![0.0f32; MAX_LEN * D];
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let mut scores = vec![0.0f32; MAX_LEN];
        for head in 0..N_HEADS {
            let base = head * MAX_LEN * HEAD_DIM;
            for t in 0..MAX_LEN {
                for s in 0..MAX_LEN {
                    let mut dot = 0.0f32;
                    for j in 0..HEAD_DIM {
                        dot += q[base + t * HEAD_DIM + j] * k[base + s * HEAD_DIM + j];
                    }
                    scores[s] = dot * scale;
                }
                softmax(&mut scores);
                for s in 0..MAX_LEN {
                    let p = scores[s];
                    for j in 0..HEAD_DIM {
                        attn_out[t * D + head * HEAD_DIM + j] += p * v[base + s * HEAD_DIM + j];
                    }
                }
            }
        }
        let projected = self.out_proj.forward(&attn_out);
        for i in 0..x.len() {
            x[i] += projected[i];
        }
        h = x.clone();
        self.norm2.forward(&mut h);
        self.ffn.forward(x, &h);
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
            final_norm: Norm::load("final_norm"),
            head0: Linear::load("head.0"),
            head3: Linear::load("head.3"),
            rope: Rope::load(),
        }
    }

    pub fn predict(&self, byte_ids: &[usize]) -> Vec<f32> {
        let mut x = vec![0.0f32; MAX_LEN * D];
        for pos in 0..MAX_LEN {
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
            let gram = &self.ngram_embed[bucket * NGRAM_DIM..bucket * NGRAM_DIM + NGRAM_DIM];
            for o in 0..D {
                let row = &self.ngram_proj.weight[o * NGRAM_DIM..o * NGRAM_DIM + NGRAM_DIM];
                let mut sum = 0.0f32;
                for i in 0..NGRAM_DIM {
                    sum += row[i] * gram[i];
                }
                x[pos * D + o] += sum;
            }
        }

        for layer in &self.conv_layers {
            layer.forward(&mut x);
        }

        for layer in &self.attn_layers {
            layer.forward(&mut x, &self.rope);
        }
        self.final_norm.forward(&mut x);
        let mut pooled = vec![0.0f32; D];
        let mut count = 0usize;
        for pos in 0..MAX_LEN {
            if byte_ids[pos] != PAD {
                count += 1;
                for d in 0..D {
                    pooled[d] += x[pos * D + d];
                }
            }
        }
        let denom = count.max(1) as f32;
        for value in &mut pooled {
            *value /= denom;
        }

        let hidden = self.head0.forward(&pooled);
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

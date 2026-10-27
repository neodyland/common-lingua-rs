pub const D: usize = 256;
pub const N_CONV: usize = 3;
pub const N_ATTN: usize = 1;
pub const N_HEADS: usize = 4;
pub const HEAD_DIM: usize = 64;
pub const HALF: usize = HEAD_DIM / 2;
pub const KERNEL: usize = 15;
pub const NGRAM_BUCKETS: usize = 4096;
pub const NGRAM_DIM: usize = 64;
pub static MAX_LEN: usize = 512;
pub const NUM_CLASSES: usize = 334;
pub const FFN: usize = 512;
pub const PAD: usize = 256;
pub const Q_SCALE: f32 = 0.125;

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

pub fn encode(text: &str) -> Vec<usize> {
    let mut ids = vec![PAD; MAX_LEN];
    let raw = text.as_bytes();
    let len = raw.len().min(MAX_LEN);
    for i in 0..len {
        ids[i] = raw[i] as usize;
    }
    ids
}

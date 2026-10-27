use common_lingua::{LANGS, MAX_LEN, Model, encode};

fn main() {
    let model = Model::load();
    let texts: Vec<String> = std::env::args().skip(1).collect();
    println!(
        "# {} languages, max_len={}, dtype=fp32, device=cpu",
        LANGS.len(),
        MAX_LEN
    );
    for text in &texts {
        let probs = model.predict(&encode(text));
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let preview: String = text.chars().take(80).collect::<String>().replace('\n', " ");
        let others = order[1..3]
            .iter()
            .map(|&i| format!("{}={:.3}", LANGS[i], probs[i]))
            .collect::<Vec<String>>()
            .join(" ");
        println!(
            "{}\t{:.4}\t{}\t{}",
            LANGS[order[0]], probs[order[0]], others, preview
        );
    }
}

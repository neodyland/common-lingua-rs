use common_lingua::{LANGS, MAX_LEN, Model, encode};

fn main() {
    let mut model = Model::load(include_bytes!("./weight.safetensors")).unwrap();
    model.tune();
    println!("selected profile: {}", model.profile);
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
        let top = order[0];
        let others: f32 = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != top)
            .map(|(_, p)| *p)
            .sum();
        println!(
            "{}\t{:.4}\t{:.3} others\t{}",
            LANGS[top], probs[top], others, preview
        );
    }
}

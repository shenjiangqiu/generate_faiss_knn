use std::path::PathBuf;

use clap::Parser;
use generate_faiss_knn::{init_logger_info, read_fvecs::{Fvec, Ivec}};
use tracing::info;


#[derive(Debug, Parser)]
struct Cli {
    // train: PathBuf,
    base: PathBuf,
    query: PathBuf,
    k: usize,
    save: PathBuf,
}



fn main() {
    init_logger_info();
    info!("generate the ground truth");
    let cli = Cli::parse();
    info!("{:?}", cli);
    // let train = Fvec::from_file(&cli.train);
    info!("load base and query");
    let base = Fvec::from_file(&cli.base);
    let query = Fvec::from_file(&cli.query);
    info!("compute the ground truth");
    let ground_true = generate_faiss_knn::ground_true(&base, &query, cli.k);
    let ground_truth = ground_true
        .into_iter()
        .flatten()
        .map(|x| x.index as u32)
        .collect();
    let ivecs = Ivec::new(cli.k, query.num, ground_truth);
    info!("save the ground truth");
    ivecs.save(&cli.save);
    // save it to a file
}

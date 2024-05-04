use std::path::PathBuf;

use clap::Parser;
use generate_faiss_knn::{
    init_logger_info,
    read_fvecs::{Fvec, Ivec},
};
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

    // check same results
    for query_id in 0..(10.min(ivecs.num)) {
        let query_result = ivecs.get_node(query_id);
        assert_eq!(query_result.len(), cli.k);
        info!("testing topK: {:?}", query_result);
        for i in query_result.into_iter() {
            let base_vec = base.get_node(*i as usize);
            let query_vec = query.get_node(query_id);
            let distance = generate_faiss_knn::distance(query_vec, base_vec, base.dim);
            info!("distance: {:?}", distance);
        }
    }

    ivecs.save(&cli.save);
    // save it to a file
}

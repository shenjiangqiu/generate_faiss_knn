use std::path::PathBuf;

use clap::Parser;
use generate_faiss_knn::read_fvecs::Fvec;

#[derive(Debug, Parser)]
struct Cli {
    file: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let (dim, num) = Fvec::read_size(&cli.file);
    println!("dim: {}, num: {}", dim, num);
}

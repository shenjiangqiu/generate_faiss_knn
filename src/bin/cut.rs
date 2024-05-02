use std::path::PathBuf;

use clap::Parser;
use generate_faiss_knn::read_fvecs::Fvec;

#[derive(Debug, Parser)]
struct Cli {
    file: PathBuf,
    size: usize,
    save: PathBuf,
}
fn main() {
    let cli = Cli::parse();
    let old_fvec = Fvec::from_file_slice(&cli.file, 0, cli.size);
    old_fvec.save(&cli.save);
}

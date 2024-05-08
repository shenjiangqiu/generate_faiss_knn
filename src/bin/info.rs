use std::path::PathBuf;

use clap::Parser;
use generate_faiss_knn::read_fvecs::{Fvec, Ivec};

#[derive(Debug, Parser)]
struct Cli {
    file: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let extension = cli.file.extension().unwrap().to_str().unwrap();
    let (dim, num) = match extension {
        "ivecs" => {
            println!("ivecs");
            Ivec::read_size(&cli.file)
        }
        "fvecs" => {
            println!("fvecs");
            Fvec::read_size(&cli.file)
        }
        _ => {
            panic!("unknown file extension: {}", extension);
        }
    };
    println!("dim: {}, num: {}", dim, num);
}

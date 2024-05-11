use std::path::{Path, PathBuf};

use clap::Parser;
use generate_faiss_knn::read_fvecs::{Fvec, Ivec};

#[derive(Debug, Parser)]
struct Cli {
    file: PathBuf,
}

fn explore_dir(path: &Path) {
    let files = std::fs::read_dir(path).unwrap();
    for f in files {
        let path = f.unwrap().path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_str().unwrap();
                if ext == "fvecs" || ext == "ivecs" {
                    println!("{}", path.display());
                    explore(&path);
                }
            }
        } else {
            explore_dir(&path);
        }
    }
}

fn explore(path: &Path) {
    let extension = path.extension().unwrap().to_str().unwrap();
    let (dim, num) = match extension {
        "ivecs" => {
            println!("ivecs");
            Ivec::read_size(path)
        }
        "fvecs" => {
            println!("fvecs");
            Fvec::read_size(path)
        }
        _ => {
            panic!("unknown file extension: {}", extension);
        }
    };
    println!("dim: {}, num: {}", dim, num);
}
fn main() {
    let cli = Cli::parse();
    if cli.file.is_dir() {
        explore_dir(&cli.file);
    } else {
        explore(&cli.file);
    }
}

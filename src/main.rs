use std::path::Path;

use generate_faiss_knn::read_fvecs::Fvec;

const DATA_PATH: &str = "data/oxford5k/oxc1_hesaff_sift.bin";
fn main() {
    println!("Hello, world!");
    let _data = Fvec::from_file(Path::new(DATA_PATH));
}

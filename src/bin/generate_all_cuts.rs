// export const datasets = [
//     {name:"deep",path:"../dataset/deep",origin:"deep_100M_base.fvecs"}
//     {name:"sift",path:"../dataset/sift_100M",origin:"sift_100M.fvecs"}
//     {name:"spaceev",path:"../dataset/spaceev",origin:"spacev100m_base.fvecs"}
// ]

use std::path::PathBuf;

use generate_faiss_knn::{
    init_logger_info,
    read_fvecs::{Fvec, Ivec},
};
use tracing::info;

struct DatasetEntry {
    name: String,
    path: PathBuf,
}

fn main() {
    init_logger_info();
    let datasets = vec![
        DatasetEntry {
            name: "deep".to_string(),
            path: PathBuf::from("dataset/deep"),
        },
        DatasetEntry {
            name: "sift".to_string(),
            path: PathBuf::from("dataset/sift_100M"),
        },
        DatasetEntry {
            name: "spaceev".to_string(),
            path: PathBuf::from("dataset/spaceev"),
        },
    ];

    for d in datasets {
        info!("building: {}", d.name);
        let origin_path = d.path.join("base_100M.fvecs");
        let base_path = d.path.join("base_10M.fvecs");

        let old_train_path = d.path.join("train_1M.fvecs");
        let train_path = d.path.join("train_100K.fvecs");

        let old_query_path = d.path.join("query_100K.fvecs");
        let query_path = d.path.join("query_10K.fvecs");

        let old_gt_path = d.path.join("gt_100K.ivecs");
        let gt_path = d.path.join("gt_10K.ivecs");

        // first cut the base
        if !base_path.exists() {
            info!("building: {}", base_path.display());
            Fvec::from_file_slice(&origin_path, 0, 10000000).save(&base_path);
        } else {
            info!("skip: {}", base_path.display());
        }
        // cut the train
        if !train_path.exists() {
            info!("building: {}", train_path.display());
            Fvec::from_file_slice(&old_train_path, 0, 100000).save(&train_path);
        } else {
            info!("skip: {}", train_path.display());
        }
        // cut the query
        if !query_path.exists() {
            info!("building: {}", query_path.display());
            Fvec::from_file_slice(&old_query_path, 0, 10000).save(&query_path);
        } else {
            info!("skip: {}", query_path.display());
        }
        // cut the gt
        if !gt_path.exists() {
            info!("building: {}", gt_path.display());
            Ivec::from_file_slice(&old_gt_path, 0, 10000).save(&gt_path);
        } else {
            info!("skip: {}", gt_path.display());
        }
    }
}

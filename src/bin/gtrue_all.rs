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
const K: usize = 100;
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
        // let origin_path = d.path.join("base_100M.fvecs");
        let base_path = d.path.join("base_10M.fvecs");

        // let old_train_path = d.path.join("train_1M.fvecs");
        // let train_path = d.path.join("train_100K.fvecs");

        // let old_query_path = d.path.join("query_100K.fvecs");
        let query_path = d.path.join("query_10K.fvecs");

        // let old_gt_path = d.path.join("gt_100K.ivecs");
        let gt_path = d.path.join("gt_10K.ivecs");

        info!("generate the ground truth");
        // let train = Fvec::from_file(&cli.train);
        info!("load base and query");
        let base = Fvec::from_file(&base_path);
        let query = Fvec::from_file(&query_path);
        info!("compute the ground truth");
        let ground_true = generate_faiss_knn::ground_true(&base, &query, K);
        let ground_truth = ground_true
            .into_iter()
            .flatten()
            .map(|x| x.index as u32)
            .collect();
        let ivecs = Ivec::new(K, query.num, ground_truth);
        info!("save the ground truth");

        // check same results
        for query_id in 0..(10.min(ivecs.num)) {
            let query_result = ivecs.get_node(query_id);
            assert_eq!(query_result.len(), K);
            info!("testing topK: {:?}", query_result);
            for i in query_result.into_iter() {
                let base_vec = base.get_node(*i as usize);
                let query_vec = query.get_node(query_id);
                let distance = generate_faiss_knn::distance(query_vec, base_vec, base.dim);
                info!("distance: {:?}", distance);
            }
        }

        ivecs.save(&gt_path);
        // cut the gt
        // if !gt_path.exists() {
        // info!("building: {}", gt_path.display());
        // Ivec::from_file_slice(&old_gt_path, 0, 10000).save(&gt_path);
        // } else {
        // info!("skip: {}", gt_path.display());
        // }
    }
}

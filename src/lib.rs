use std::{cmp::Reverse, collections::BinaryHeap, sync::atomic::AtomicUsize};

use read_fvecs::Fvec;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::EnvFilter;

pub mod read_fvecs;

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        pub fn init_logger_info();
        pub fn info_str(s: &str);
    }

}
fn info_str(s:&str){
    info!("{}",s);
}

pub fn init_logger_info() {
    tracing_subscriber::fmt::SubscriberBuilder::default()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .try_init()
        .ok();
}
pub fn distance(node: &[f32], base: &[f32], dim: usize) -> Vec<f32> {
    assert!(node.len() == dim);
    assert!(base.len() % dim == 0);
    let base_num = base.len() / dim;
    let mut distances = Vec::with_capacity(base_num);
    for i in 0..base_num {
        let mut distance = 0f32;
        for j in 0..dim {
            distance += (node[j] - base[i * dim + j]).powi(2);
        }
        distances.push(distance.sqrt());
    }
    distances
}
#[derive(Debug, PartialEq, PartialOrd)]
pub struct DistanceWithIndex {
    pub distance: f32,
    pub index: usize,
}

impl Eq for DistanceWithIndex {}

impl Ord for DistanceWithIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}
pub fn ground_true(base: &Fvec, query: &Fvec, k: usize) -> Vec<Vec<DistanceWithIndex>> {
    assert_eq!(base.dim, query.dim);
    use rayon::prelude::*;
    let remaining_jobs = AtomicUsize::new(query.num);
    let ground_true = (0..query.num)
        .into_par_iter()
        .map(|node_id| {
            let node = query.get_node(node_id);
            // compute the l2 distance of node to all base nodes
            let distances = distance(node, &base.data, base.dim);
            let distances_with_index = distances
                .iter()
                .enumerate()
                .map(|(index, &distance)| Reverse(DistanceWithIndex { distance, index }))
                .collect::<Vec<_>>();
            let mut bin_heap = BinaryHeap::from(distances_with_index);
            let mut knn = Vec::with_capacity(k);
            for _ in 0..k {
                if let Some(Reverse(dist)) = bin_heap.pop() {
                    knn.push(dist);
                }
            }
            assert!(knn.len() == k);
            let previouse = remaining_jobs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            info!("remaining jobs: {}/{}", previouse - 1, query.num);
            knn
        })
        .collect();
    ground_true
}

#[cfg(test)]
mod tests {
    use crate::read_fvecs::Fvec;

    #[test]
    fn test_ground_true() {
        let base = Fvec::new(
            4,
            10,
            (0..10)
                .into_iter()
                .map(|x| [x as f32; 4])
                .flatten()
                .collect(),
        );
        let query = Fvec::new(
            4,
            2,
            (3..5)
                .into_iter()
                .map(|x| [x as f32; 4])
                .flatten()
                .collect(),
        );
        let ground_true = super::ground_true(&base, &query, 3);
        assert_eq!(ground_true.len(), 2);
        assert_eq!(ground_true[0].len(), 3);
        println!("{:?}", ground_true);
    }
}

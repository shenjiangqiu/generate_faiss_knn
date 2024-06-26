# cargo run --bin gtrue --release -- ./dataset/deep/deep_100M_base.fvecs ./dataset/deep/deep_100M_query_100K.fvecs 100 ./dataset/deep/deep_100M_gt.ivecs
# cargo run --bin gtrue --release -- dataset/sift_100M/sift_100M.fvecs dataset/sift_100M/sift_100M_query_100K.fvecs 100 dataset/sift_100M/sift_100M_gt.fvecs.ivecs
# cargo run --bin gtrue --release -- dataset/spacev/spacev100m_base.fvecs dataset/spacev/spaceev_100M_query_100K.fvecs 100 dataset/spacev/spaceev_100M_gt.ivecs

cargo run --bin gtrue --release -- ./dataset/deep/deep_100M_query_100K.fvecs ./dataset/deep/deep_100M_test_10.fvecs 100 ./dataset/deep/deep_100M_gt_test_10.ivecs

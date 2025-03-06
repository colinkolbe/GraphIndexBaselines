
use graphidx::indices::KnnIndex;
use graphidx::measures::SquaredEuclideanDistance;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::prelude::Distribution;

use criterion::{criterion_group, criterion_main, Criterion};
use paste::paste;


macro_rules! my_criterion_group {
	($name: ident, $($bench: ident $([$n_samples: literal])?),* $(,)?) => {
		paste! {
			$(
				fn [<$bench _wrap>](c: &mut Criterion) {
					use std::time::Duration;
					let mut c = c.benchmark_group(stringify!($bench));
					c.measurement_time(Duration::new(10_000, 0));
					c.sample_size(100);
					$(
						c.sample_size($n_samples);
					)?
					c.bench_function(stringify!($bench), |_| $bench());
				}
			)*
			criterion_group!($name, $([<$bench _wrap>]),*);
		}
	};
}

fn init_data() -> (Array2<f64>, Array2<f64>) {
	let (n_data, n_queries, d) = (100_000, 10_000, 20);
	let rng = Normal::new(0.0, 1.0).unwrap();
	let data: Array2<f64> = Array2::from_shape_fn((n_data, d), |_| rng.sample(&mut rand::thread_rng()));
	let queries: Array2<f64> = Array2::from_shape_fn((n_queries, d), |_| rng.sample(&mut rand::thread_rng()));
	(data, queries)
}

fn hnsw_construction() {
	use graphidxbaselines::hnsw::*;
	let (data, queries) = init_data();
	let params = HNSWParams::new()
	.with_insert_heuristic(true)
	.with_insert_heuristic_extend(false)
	;
	let index = HNSWParallelHeapBuilder::<u64,_,_>::build(data.view(), SquaredEuclideanDistance::new(), params, 1);
	let _ = index.knn_query_batch(&queries, 100);
}
fn hnsw_capped_construction() {
	use graphidxbaselines::hnsw::*;
	let (data, queries) = init_data();
	let params = HNSWParams::new()
	.with_insert_heuristic(true)
	.with_insert_heuristic_extend(false)
	.with_max_build_frontier_size(Some(10))
	;
	let index = HNSWParallelHeapBuilder::<u64,_,_>::build_capped(data.view(), SquaredEuclideanDistance::new(), params, 1, 10);
	let _ = index.knn_query_batch(&queries, 100);
}
fn rnn_construction() {
	use graphidxbaselines::rnn::*;
	let (data, queries) = init_data();
	let params = RNNParams::new()
	.with_n_outer_loops(2)
	.with_n_inner_loops(10)
	;
	let index = RNNDescentBuilder::<u64,_,_>::build(data.view(), SquaredEuclideanDistance::new(), params);
	let _ = index.knn_query_batch(&queries, 100);
}
fn sen_construction() {
	use graphidxbaselines::rnn::*;
	let (data, queries) = init_data();
	let params = SENParams::new()
	.with_n_outer_loops(2)
	.with_n_inner_loops(10)
	;
	let index = SENDescentBuilder::<u64,_,_>::build(data.view(), SquaredEuclideanDistance::new(), params);
	let _ = index.knn_query_batch(&queries, 100);
}

// my_criterion_group!(
// 	benches,
// 	hnsw_construction[10],
// 	hnsw_capped_construction[10],
// 	rnn_construction[10],
// 	sen_construction[10],
// );
// criterion_main!(benches);

macro_rules! easy_bench {
	($fun: ident) => {
		let start_time = std::time::Instant::now();
		$fun();
		let elapsed_time = start_time.elapsed();
		println!("{:}: {:?}", stringify!($fun), elapsed_time);
	}
}
fn main() {
	easy_bench!(hnsw_construction);
	easy_bench!(hnsw_capped_construction);
	easy_bench!(rnn_construction);
	easy_bench!(sen_construction);
}

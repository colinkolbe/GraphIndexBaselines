use std::marker::PhantomData;

use graphidx::indices::bruteforce_neighbors;
use graphidx::measures::Distance;
use graphidx::types::*;
use graphidx::graphs::*;
use graphidx::indices::*;
use ndarray::Axis;
use ndarray::{ArrayBase,Ix2,Data};



param_struct!(BruteforceKNNParams[Copy, Clone] {
	degree: usize = 10,
	batch_size: usize = 1000,
});
pub struct BruteforceKNNGraphBuilder<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync> {
	_phantom: PhantomData<(R, F)>,
	graph: DirLoLGraph<R>,
	dist: Dist,
}
impl<R: SyncUnsignedInteger, F: SyncFloat, Dist: Distance<F>+Sync> BruteforceKNNGraphBuilder<R, F, Dist> {
	fn base_init<D: Data<Elem=F>+Sync>(mat: &ArrayBase<D, Ix2>, dist: Dist, params: BruteforceKNNParams) -> Self {
		let mut graph: DirLoLGraph<R> = DirLoLGraph::new();
		let n_data = mat.nrows();
		graph.reserve(n_data);
		(0..n_data).for_each(|_| graph.add_node());
		mat.axis_chunks_iter(Axis(0), params.batch_size)
		.enumerate().for_each(|(i_chunk,chunk)| {
			let start = i_chunk * params.batch_size;
			let end = start + chunk.len_of(Axis(0));
			let knn_ids = bruteforce_neighbors::<R,F,_,_,_>(mat, &chunk, &dist, params.degree+1).0;
			(start..end).zip(knn_ids.axis_iter(Axis(0))).for_each(|(u,vs)| {
				let u = R::from_usize(u).unwrap();
				graph.add_edges_chunk(u, &vs.into_iter().cloned().filter(|&v| u!=v).collect());
			});
		});
		Self {
			_phantom: PhantomData,
			graph,
			dist,
		}
	}
	#[inline(always)]
	pub fn build<D: Data<Elem=F>+Sync>(mat: ArrayBase<D, Ix2>, dist: Dist, params: BruteforceKNNParams) -> GreedySingleGraphIndex<R, F, Dist, ArrayBase<D, Ix2>, DirLoLGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		GreedySingleGraphIndex::new(mat, builder.graph, builder.dist, None)
	}
	#[inline(always)]
	pub fn build_capped<D: Data<Elem=F>+Sync>(mat: ArrayBase<D, Ix2>, dist: Dist, params: BruteforceKNNParams, max_frontier_size: usize) -> GreedyCappedSingleGraphIndex<R, F, Dist, ArrayBase<D, Ix2>, DirLoLGraph<R>> {
		let builder = Self::base_init(&mat, dist, params);
		GreedyCappedSingleGraphIndex::new(mat, builder.graph, builder.dist, max_frontier_size, None)
	}
}


#[cfg(test)]
mod tests {
	use crate::knn::*;
	use ndarray::{Array2,Slice};
	use ndarray_rand::rand_distr::Normal;
	use rand::prelude::Distribution;
	use ndarray::Axis;
	use std::time::Instant;
	use rand::distributions::Uniform;

	use graphidx::measures::*;


	#[test]
	fn knn_probs_test() {
		/* Limit global thread pool size */
		// rayon::ThreadPoolBuilder::new().num_threads(16).build_global().unwrap();
		/* Parameters */
		let (nd, nq, d, k, search_heap_size) = (100_000, 1000, 50, 1, 10);
		let euc = SquaredEuclideanDistance::new();
		/* Data initialization */
		let init_time = Instant::now();
		type R = usize;
		type F = f32;
		type Dist = SquaredEuclideanDistance<F>;
		let data: Array2<F> = if 0>0 { /* Normal distribution */
			let rng = Normal::new(0.0, 1.0).unwrap();
			Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
		} else { /* Uniform distribution */
			let rng = Uniform::new(0.0, 1.0);
			Array2::from_shape_fn((nd, d), |_| rng.sample(&mut rand::thread_rng()))
		};
		let queries = data.slice_axis(Axis(0), Slice::from(0..nq));
		println!("Data initialization: {:.2?}", init_time.elapsed());
		/* Build and translate RNN Graph */
		let build_time = Instant::now();
		let dist = Dist::new();
		let params = BruteforceKNNParams::new().with_degree(145);
		type BuilderType = BruteforceKNNGraphBuilder<R,F,Dist>;
		let index1 = BuilderType::build(data.view(), dist, params);
		println!("Graph construction ({:?}): {:.2?}", std::any::type_name::<BuilderType>(), build_time.elapsed());
		let index2 = GreedyCappedSingleGraphIndex::new(
			data.view(),
			index1.graph().as_dir_lol_graph(),
			SquaredEuclideanDistance::new(),
			3*k,
			None,
		);
		/* Brute force queries */
		let bruteforce_time = Instant::now();
		let (bruteforce_ids, _) = bruteforce_neighbors(&data, &queries, &euc, k);
		println!("Brute force queries: {:.2?}", bruteforce_time.elapsed());
		/* HNSW queries */
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids1, hnsw_dists1) = index1.greedy_search_batch(&queries, k, search_heap_size);
		println!("HNSW queries 1: {:.2?}", hnsw_time.elapsed());
		let hnsw_time = Instant::now();
		#[allow(unused)]
		let (hnsw_ids2, hnsw_dists2) = index2.greedy_search_batch(&queries, k, search_heap_size);
		println!("HNSW queries 2: {:.2?}", hnsw_time.elapsed());
		/* Compute and print recall */
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids1.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 1: {:.2}%", recall*100f32);
		let mut same = 0;
		bruteforce_ids.axis_iter(Axis(0)).zip(hnsw_ids2.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
			let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
			let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
			same += bf_set.intersection(&hnsw_set).count();
		});
		let recall = same as f32 / (nq * k) as f32;
		println!("Recall 2: {:.2}%", recall*100f32);

		/* Recall over NNs */
		let base_graph = index1.graph().as_viewable_adj_graph().unwrap();
		let max_degree = base_graph.view_neighbors(0).len();
		(1..=max_degree).for_each(|degree| {
			let mut graph: DirLoLGraph<R> = DirLoLGraph::new();
			(0..nd).for_each(|_| graph.add_node());
			(0..nd).for_each(|u| {
				let neighbors = base_graph.view_neighbors(u);
				graph.add_edges_chunk(u, &neighbors[..degree].iter().cloned().collect());
			});
			let index = GreedySingleGraphIndex::new(data.view(), graph, SquaredEuclideanDistance::new(), None);
			let ids = index.greedy_search_batch(&queries, k, search_heap_size).0;
			let mut same = 0;
			bruteforce_ids.axis_iter(Axis(0)).zip(ids.axis_iter(Axis(0))).for_each(|(bf, rnn)| {
				let bf_set = bf.iter().collect::<std::collections::HashSet<_>>();
				let hnsw_set = rnn.iter().collect::<std::collections::HashSet<_>>();
				same += bf_set.intersection(&hnsw_set).count();
			});
			let recall = same as f32 / (nq * k) as f32;
			println!("{:03} {:6.2}%", max_degree, recall*100f32);
		});
	}
}

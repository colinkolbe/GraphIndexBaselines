

// struct PyArrayDataSource

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use graphidx::{graphs::{DirLoLGraph, Graph}, indices::{GraphIndex, GreedyCappedLayeredGraphIndex, GreedyCappedSingleGraphIndex, GreedyLayeredGraphIndex, GreedySingleGraphIndex}, measures::SquaredEuclideanDistance, types::UnsignedInteger};
use pyo3::prelude::*;

use crate::{hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder}, rnn::RNNStyleBuilder};

/* Conversion code to handle different ndarray versions in this crate and numpy dependencies */
fn arr1_rust_to_py<T>(arr: Array1<T>) -> numpy::ndarray::Array1<T> {
	numpy::ndarray::Array1::from_vec(arr.into_raw_vec())
}
fn arr2_rust_to_py<T>(arr: Array2<T>) -> numpy::ndarray::Array2<T> {
	let shape = arr.shape();
	unsafe{numpy::ndarray::Array2::from_shape_vec_unchecked((shape[0],shape[1]), arr.into_raw_vec())}
}
fn arrview1_py_to_rust<T>(arr: numpy::ndarray::ArrayView1<T>) -> ArrayView1<'static, T> {
	let shape = arr.shape();
	unsafe{ArrayView1::from_shape_ptr((shape[0],), arr.as_ptr() as *const T)}
}
fn arrview2_py_to_rust<T>(arr: numpy::ndarray::ArrayView2<T>) -> ArrayView2<'static, T> {
	let shape = arr.shape();
	unsafe{ArrayView2::from_shape_ptr((shape[0],shape[1]), arr.as_ptr() as *const T)}
}

#[pyclass]
pub struct GraphStats {
	#[pyo3(get)]
	pub n_nodes: usize,
	#[pyo3(get)]
	pub n_edges: usize,
	#[pyo3(get)]
	pub max_degree: usize,
	#[pyo3(get)]
	pub min_degree: usize,
	#[pyo3(get)]
	pub avg_degree: f64,
	#[pyo3(get)]
	pub std_degree: f64,
}
impl GraphStats {
	pub fn from_graph<R: UnsignedInteger, G: Graph<R>>(g: &G) -> Self {
		let n_nodes = g.n_vertices();
		let (mut n_edges, mut max_degree, mut min_degree, mut ssq_degree) = (0, 0, usize::MAX, 0);
		(0..n_nodes).for_each(|i| {
			let degree = g.degree(R::from(i).unwrap());
			n_edges += degree;
			max_degree = max_degree.max(degree);
			min_degree = min_degree.min(degree);
			ssq_degree += degree*degree;
		});
		let mssq_degree = ssq_degree as f64 / n_nodes as f64;
		let avg_degree = n_edges as f64 / n_nodes as f64;
		let var_degree = mssq_degree - avg_degree*avg_degree;
		let std_degree = var_degree.sqrt();
		Self {
			n_nodes: n_nodes,
			n_edges: n_edges,
			max_degree: max_degree,
			min_degree: min_degree,
			avg_degree: avg_degree,
			std_degree: std_degree,
		}
	}
}

#[pyclass]
pub struct PyHNSW {
	index: Option<GreedyLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		ArrayView2<'static, f32>,
		DirLoLGraph<usize>,
	>>,
	capped_index: Option<GreedyCappedLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		ArrayView2<'static, f32>,
		DirLoLGraph<usize>,
	>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				1,
			);
			if max_frontier_size.is_some() {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PyHNSW { index: None, capped_index: Some(capped_index), max_frontier_size: max_frontier_size }
			} else {
				PyHNSW { index: Some(index), capped_index: None, max_frontier_size: None }
			}
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			}
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			}
		}
	}
	#[getter]
	fn get_max_frontier_size(&self) -> Option<usize> {
		self.max_frontier_size
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
		self.max_frontier_size = max_frontier_size;
		if self.max_frontier_size.is_none() && self.index.is_none() {
			let mut capped_index = None;
			std::mem::swap(&mut self.capped_index, &mut capped_index);
			let mut uncapped_index = Some(capped_index.unwrap().into_uncapped());
			std::mem::swap(&mut self.index, &mut uncapped_index);
		} else if self.max_frontier_size.is_some() && self.capped_index.is_none() {
			let mut uncapped_index = None;
			std::mem::swap(&mut self.index, &mut uncapped_index);
			let mut capped_index = Some(uncapped_index.unwrap().into_capped(self.max_frontier_size.unwrap()));
			std::mem::swap(&mut self.capped_index, &mut capped_index);
		}
		if self.max_frontier_size.is_some() {
			self.capped_index.as_mut().unwrap().max_frontier_size = self.max_frontier_size.unwrap();
		}
	}
	fn get_graph_stats(&self) -> Vec<GraphStats> {
		if self.index.is_some() {
			self.index.as_ref().unwrap().graphs().iter().map(|g| GraphStats::from_graph(g)).collect()
		} else {
			self.capped_index.as_ref().unwrap().graphs().iter().map(|g| GraphStats::from_graph(g)).collect()
		}
	}
}


#[pyclass]
pub struct OwningPyHNSW {
	index: Option<GreedyLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		Array2<f32>,
		DirLoLGraph<usize>,
	>>,
	capped_index: Option<GreedyCappedLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		Array2<f32>,
		DirLoLGraph<usize>,
	>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl OwningPyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, post_prune_heuristic=None, insert_minibatch_size=None, n_rounds=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		post_prune_heuristic: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let hnsw_params = HNSWParams::new()
		.maybe_with_higher_max_degree(higher_max_degree)
		.maybe_with_lowest_max_degree(lowest_max_degree)
		.maybe_with_max_layers(max_layers)
		.maybe_with_n_parallel_burnin(n_parallel_burnin)
		.maybe_with_max_build_heap_size(max_build_heap_size)
		.with_max_build_frontier_size(max_build_frontier_size)
		.with_level_norm_param_override(level_norm_param_override)
		.maybe_with_insert_heuristic(insert_heuristic)
		.maybe_with_insert_heuristic_extend(insert_heuristic_extend)
		.maybe_with_post_prune_heuristic(post_prune_heuristic)
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				1,
			);
			if max_frontier_size.is_some() {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				OwningPyHNSW { index: None, capped_index: Some(capped_index), max_frontier_size: max_frontier_size }
			} else {
				OwningPyHNSW { index: Some(index), capped_index: None, max_frontier_size: None }
			}
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			}
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			}
		}
	}
	#[getter]
	fn get_max_frontier_size(&self) -> Option<usize> {
		self.max_frontier_size
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
		self.max_frontier_size = max_frontier_size;
		if self.max_frontier_size.is_none() && self.index.is_none() {
			let mut capped_index = None;
			std::mem::swap(&mut self.capped_index, &mut capped_index);
			let mut uncapped_index = Some(capped_index.unwrap().into_uncapped());
			std::mem::swap(&mut self.index, &mut uncapped_index);
		} else if self.max_frontier_size.is_some() && self.capped_index.is_none() {
			let mut uncapped_index = None;
			std::mem::swap(&mut self.index, &mut uncapped_index);
			let mut capped_index = Some(uncapped_index.unwrap().into_capped(self.max_frontier_size.unwrap()));
			std::mem::swap(&mut self.capped_index, &mut capped_index);
		}
		if self.max_frontier_size.is_some() {
			self.capped_index.as_mut().unwrap().max_frontier_size = self.max_frontier_size.unwrap();
		}
	}
	fn get_graph_stats(&self) -> Vec<GraphStats> {
		if self.index.is_some() {
			self.index.as_ref().unwrap().graphs().iter().map(|g| GraphStats::from_graph(g)).collect()
		} else {
			self.capped_index.as_ref().unwrap().graphs().iter().map(|g| GraphStats::from_graph(g)).collect()
		}
	}
}

#[pyfunction]
#[pyo3(signature = (file, max_frontier_size=None))]
pub fn load_hnswlib(file: &str, max_frontier_size: Option<usize>) -> OwningPyHNSW {
	let index = crate::hnsw::load_hnswlib(file);
	if max_frontier_size.is_none() {
		OwningPyHNSW{index:Some(index), capped_index:None, max_frontier_size:None}
	} else {
		OwningPyHNSW{index:None, capped_index:Some(index.into_capped(max_frontier_size.unwrap())), max_frontier_size:max_frontier_size}
	}
}

#[pyclass]
pub struct PyRNNDescent {
	index: Option<GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, ArrayView2<'static,f32>, DirLoLGraph<usize>>>,
	capped_index: Option<GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, ArrayView2<'static,f32>, DirLoLGraph<usize>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PyRNNDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let params = crate::rnn::RNNParams::new()
		.maybe_with_initial_degree(initial_degree)
		.maybe_with_reduce_degree(reduce_degree)
		.maybe_with_n_outer_loops(n_outer_loops)
		.maybe_with_n_inner_loops(n_inner_loops)
		.maybe_with_concurrent_batch_size(concurrent_batch_size)
		;
		unsafe {
			let index = crate::rnn::RNNDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				params,
			);
			if max_frontier_size.is_none() {
				PyRNNDescent { index: Some(index), capped_index: None, max_frontier_size: None }
			} else {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PyRNNDescent { index: None, capped_index: Some(capped_index), max_frontier_size: max_frontier_size }
			}
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			}
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			}
		}
	}
	#[getter]
	fn get_max_frontier_size(&self) -> Option<usize> {
		self.max_frontier_size
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
		self.max_frontier_size = max_frontier_size;
		if self.max_frontier_size.is_none() && self.index.is_none() {
			let mut capped_index = None;
			std::mem::swap(&mut self.capped_index, &mut capped_index);
			let mut uncapped_index = Some(capped_index.unwrap().into_uncapped());
			std::mem::swap(&mut self.index, &mut uncapped_index);
		} else if self.max_frontier_size.is_some() && self.capped_index.is_none() {
			let mut uncapped_index = None;
			std::mem::swap(&mut self.index, &mut uncapped_index);
			let mut capped_index = Some(uncapped_index.unwrap().into_capped(self.max_frontier_size.unwrap()));
			std::mem::swap(&mut self.capped_index, &mut capped_index);
		}
		if self.max_frontier_size.is_some() {
			self.capped_index.as_mut().unwrap().max_frontier_size = self.max_frontier_size.unwrap();
		}
	}
	fn get_graph_stats(&self) -> GraphStats {
		if self.index.is_some() {
			GraphStats::from_graph(self.index.as_ref().unwrap().graph())
		} else {
			GraphStats::from_graph(self.capped_index.as_ref().unwrap().graph())
		}
	}
}

#[pyclass]
pub struct PySENDescent {
	index: Option<GreedySingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, ArrayView2<'static,f32>, DirLoLGraph<usize>>>,
	capped_index: Option<GreedyCappedSingleGraphIndex<usize, f32, SquaredEuclideanDistance<f32>, ArrayView2<'static,f32>, DirLoLGraph<usize>>>,
	max_frontier_size: Option<usize>,
}
#[pymethods]
impl PySENDescent {
	#[new]
	#[pyo3(signature = (data, initial_degree=None, reduce_degree=None, n_outer_loops=None, n_inner_loops=None, concurrent_batch_size=None, max_cos=None, dist_is_sq=None, prune_non_sen_edges=None, verify_sen_edges=None, max_frontier_size=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		initial_degree: Option<usize>,
		reduce_degree: Option<usize>,
		n_outer_loops: Option<usize>,
		n_inner_loops: Option<usize>,
		concurrent_batch_size: Option<usize>,
		max_cos: Option<f32>,
		dist_is_sq: Option<bool>,
		prune_non_sen_edges: Option<bool>,
		verify_sen_edges: Option<bool>,
		max_frontier_size: Option<usize>,
	) -> Self {
		let params = crate::rnn::SENParams::new()
		.maybe_with_initial_degree(initial_degree)
		.maybe_with_reduce_degree(reduce_degree)
		.maybe_with_n_outer_loops(n_outer_loops)
		.maybe_with_n_inner_loops(n_inner_loops)
		.maybe_with_concurrent_batch_size(concurrent_batch_size)
		.maybe_with_max_cos(max_cos)
		.maybe_with_dist_is_sq(dist_is_sq)
		.maybe_with_prune_non_sen_edges(prune_non_sen_edges)
		.maybe_with_verify_sen_edges(verify_sen_edges)
		;
		unsafe {
			let index = crate::rnn::SENDescentBuilder::build(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				params,
			);
			if max_frontier_size.is_none() {
				PySENDescent { index: Some(index), capped_index: None, max_frontier_size: None }
			} else {
				let capped_index = index.into_capped(max_frontier_size.unwrap_unchecked());
				PySENDescent { index: None, capped_index: Some(capped_index), max_frontier_size: max_frontier_size }
			}
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search(
					&arrview1_py_to_rust(query.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
					&mut index._new_search_cache(max_heap_size.unwrap_or(2*k)),
				);
				(
					PyArray1::from_owned_array(py, arr1_rust_to_py(ids)),
					PyArray1::from_owned_array(py, arr1_rust_to_py(dists)),
				)
			}
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		unsafe {
			if self.index.is_some() {
				let index = self.index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			} else {
				let index = self.capped_index.as_ref().unwrap_unchecked();
				let (ids, dists) = index.greedy_search_batch(
					&arrview2_py_to_rust(queries.as_array()),
					k,
					max_heap_size.unwrap_or(2*k),
				);
				(
					PyArray2::from_owned_array(py, arr2_rust_to_py(ids)),
					PyArray2::from_owned_array(py, arr2_rust_to_py(dists)),
				)
			}
		}
	}
	#[getter]
	fn get_max_frontier_size(&self) -> Option<usize> {
		self.max_frontier_size
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: Option<usize>) {
		self.max_frontier_size = max_frontier_size;
		if self.max_frontier_size.is_none() && self.index.is_none() {
			let mut capped_index = None;
			std::mem::swap(&mut self.capped_index, &mut capped_index);
			let mut uncapped_index = Some(capped_index.unwrap().into_uncapped());
			std::mem::swap(&mut self.index, &mut uncapped_index);
		} else if self.max_frontier_size.is_some() && self.capped_index.is_none() {
			let mut uncapped_index = None;
			std::mem::swap(&mut self.index, &mut uncapped_index);
			let mut capped_index = Some(uncapped_index.unwrap().into_capped(self.max_frontier_size.unwrap()));
			std::mem::swap(&mut self.capped_index, &mut capped_index);
		}
		if self.max_frontier_size.is_some() {
			self.capped_index.as_mut().unwrap().max_frontier_size = self.max_frontier_size.unwrap();
		}
	}
	fn get_graph_stats(&self) -> GraphStats {
		if self.index.is_some() {
			GraphStats::from_graph(self.index.as_ref().unwrap().graph())
		} else {
			GraphStats::from_graph(self.capped_index.as_ref().unwrap().graph())
		}
	}
}



#[pymodule(name="graphidxbaselines")]
fn hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PyHNSW>()?;
	m.add_class::<OwningPyHNSW>()?;
	m.add_class::<PyRNNDescent>()?;
	m.add_class::<PySENDescent>()?;
	m.add_function(wrap_pyfunction!(load_hnswlib, m)?)?;
	Ok(())
}



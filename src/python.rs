

// struct PyArrayDataSource

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use graphidx::{graphs::DirLoLGraph, indices::{GraphIndex, GreedyCappedLayeredGraphIndex, GreedyLayeredGraphIndex}, measures::SquaredEuclideanDistance};
use pyo3::prelude::*;

use crate::hnsw::{HNSWParallelHeapBuilder, HNSWParams, HNSWStyleBuilder};

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
pub struct PyHNSW {
	index: Option<GreedyLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		ArrayView2<'static, f32>,
		DirLoLGraph<usize>,
	>>,
}
#[pymethods]
impl PyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, insert_minibatch_size=None, n_rounds=None))]
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
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
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
			// index.make_fat();
			PyHNSW { index: Some(index) }
		}
	}
	fn valid(&self) -> bool {
		self.index.is_some()
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	fn into_capped(&mut self, max_frontier_size: usize) -> CappedPyHNSW {
		assert!(self.index.is_some(), "Index is not valid");
		let mut index = None;
		std::mem::swap(&mut self.index, &mut index);
		CappedPyHNSW { index: Some(index.unwrap().into_capped(max_frontier_size)) }
	}
}
#[pyclass]
pub struct CappedPyHNSW {
	index: Option<GreedyCappedLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		ArrayView2<'static, f32>,
		DirLoLGraph<usize>,
	>>,
}
#[pymethods]
impl CappedPyHNSW {
	#[new]
	#[pyo3(signature = (data, max_frontier_size, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, insert_minibatch_size=None, n_rounds=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		max_frontier_size: usize,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
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
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build_capped(
				arrview2_py_to_rust(data.as_array()),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				1,
				max_frontier_size,
			);
			// index.make_fat();
			CappedPyHNSW { index: Some(index) }
		}
	}
	fn valid(&self) -> bool {
		self.index.is_some()
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	fn into_uncapped(&mut self) -> PyHNSW {
		assert!(self.index.is_some(), "Index is not valid");
		let mut index = None;
		std::mem::swap(&mut self.index, &mut index);
		PyHNSW { index: Some(index.unwrap().into_uncapped()) }
	}
	#[getter]
	fn get_max_frontier_size(&self) -> usize {
		assert!(self.index.is_some(), "Index is not valid");
		self.index.as_ref().unwrap().max_frontier_size()
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: usize) {
		assert!(self.index.is_some(), "Index is not valid");
		self.index.as_mut().unwrap().set_max_frontier_size(max_frontier_size);
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
}
#[pymethods]
impl OwningPyHNSW {
	#[new]
	#[pyo3(signature = (data, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, insert_minibatch_size=None, n_rounds=None))]
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
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
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
			// index.make_fat();
			OwningPyHNSW { index: Some(index) }
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	fn into_capped(&mut self, max_frontier_size: usize) -> OwningCappedPyHNSW {
		assert!(self.index.is_some(), "Index is not valid");
		let mut index = None;
		std::mem::swap(&mut self.index, &mut index);
		OwningCappedPyHNSW { index: Some(index.unwrap().into_capped(max_frontier_size)) }
	}
}
#[pyclass]
pub struct OwningCappedPyHNSW {
	index: Option<GreedyCappedLayeredGraphIndex<
		usize,
		f32,
		SquaredEuclideanDistance<f32>,
		Array2<f32>,
		DirLoLGraph<usize>,
	>>,
}
#[pymethods]
impl OwningCappedPyHNSW {
	#[new]
	#[pyo3(signature = (data, max_frontier_size, higher_max_degree=None, lowest_max_degree=None, max_layers=None, n_parallel_burnin=None, max_build_heap_size=None, max_build_frontier_size=None, level_norm_param_override=None, insert_heuristic=None, insert_heuristic_extend=None, insert_minibatch_size=None, n_rounds=None))]
	fn new<'py>(
		data: Bound<'py, PyArray2<f32>>,
		max_frontier_size: usize,
		higher_max_degree: Option<usize>,
		lowest_max_degree: Option<usize>,
		max_layers: Option<usize>,
		n_parallel_burnin: Option<usize>,
		max_build_heap_size: Option<usize>,
		max_build_frontier_size: Option<usize>,
		level_norm_param_override: Option<f32>,
		insert_heuristic: Option<bool>,
		insert_heuristic_extend: Option<bool>,
		insert_minibatch_size: Option<usize>,
		n_rounds: Option<usize>,
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
		.maybe_with_insert_minibatch_size(insert_minibatch_size)
		.maybe_with_n_rounds(n_rounds)
		;
		unsafe {
			let index = HNSWParallelHeapBuilder::build_capped(
				arrview2_py_to_rust(data.as_array()).into_owned(),
				SquaredEuclideanDistance::new(),
				hnsw_params,
				1,
				max_frontier_size,
			);
			// index.make_fat();
			OwningCappedPyHNSW { index: Some(index) }
		}
	}
	#[pyo3(signature = (query, k, max_heap_size=None))]
	fn knn_query<'py>(&self, py: Python<'py>, query: Bound<'py, PyArray1<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray1<usize>>, Bound<'py,PyArray1<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	#[pyo3(signature = (queries, k, max_heap_size=None))]
	fn knn_query_batch<'py>(&self, py: Python<'py>, queries: Bound<'py, PyArray2<f32>>, k: usize, max_heap_size: Option<usize>) -> (Bound<'py,PyArray2<usize>>, Bound<'py,PyArray2<f32>>) {
		assert!(self.index.is_some(), "Index is not valid");
		unsafe {
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
		}
	}
	fn into_uncapped(&mut self) -> OwningPyHNSW {
		assert!(self.index.is_some(), "Index is not valid");
		let mut index = None;
		std::mem::swap(&mut self.index, &mut index);
		OwningPyHNSW { index: Some(index.unwrap().into_uncapped()) }
	}
	#[getter]
	fn get_max_frontier_size(&self) -> usize {
		assert!(self.index.is_some(), "Index is not valid");
		self.index.as_ref().unwrap().max_frontier_size()
	}
	#[setter]
	fn set_max_frontier_size(&mut self, max_frontier_size: usize) {
		assert!(self.index.is_some(), "Index is not valid");
		self.index.as_mut().unwrap().set_max_frontier_size(max_frontier_size);
	}
}

#[pyfunction]
pub fn load_hnswlib(file: &str) -> OwningPyHNSW {
	let index = crate::hnsw::load_hnswlib(file);
	// index.make_fat();
	OwningPyHNSW{index:Some(index)}
}
#[pyfunction]
pub fn load_hnswlib_capped(file: &str, max_frontier_size: usize) -> OwningCappedPyHNSW {
	let index = crate::hnsw::load_hnswlib_capped(file, max_frontier_size);
	// index.make_fat();
	OwningCappedPyHNSW{index:Some(index)}
}

#[pymodule(name="graphidxbaselines")]
fn hnsw(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PyHNSW>()?;
	m.add_class::<CappedPyHNSW>()?;
	m.add_class::<OwningPyHNSW>()?;
	m.add_class::<OwningCappedPyHNSW>()?;
	m.add_function(wrap_pyfunction!(load_hnswlib, m)?)?;
	m.add_function(wrap_pyfunction!(load_hnswlib_capped, m)?)?;
	Ok(())
}



import numpy as np
import time
import graphidxbaselines as gib

def init_data():
	n_data, n_queries, d = 100_000, 10_000, 20
	data = np.random.normal(0, 1, (n_data, d)).astype(np.float32)
	queries = np.random.normal(0, 1, (n_queries, d)).astype(np.float32)
	return data, queries

def hnsw_construction():
	data, queries = init_data()
	index = gib.PyHNSW(
		data,
		insert_heuristic=True,
		insert_heuristic_extend=False,
	)
	_ = index.knn_query_batch(queries, 100)
def hnsw_capped_construction():
	data, queries = init_data()
	index = gib.PyHNSW(
		data,
		insert_heuristic=True,
		insert_heuristic_extend=False,
		max_build_frontier_size=10,
		max_frontier_size=10,
	)
	_ = index.knn_query_batch(queries, 100)
def rnn_construction():
	data, queries = init_data()
	index = gib.PyRNNDescent(
		data,
		n_outer_loops=2,
		n_inner_loops=10,
	)
	_ = index.knn_query_batch(queries, 100)
def sen_construction():
	data, queries = init_data()
	index = gib.PyRNNDescent(
		data,
		n_outer_loops=2,
		n_inner_loops=10,
	)
	_ = index.knn_query_batch(queries, 100)

def easy_bench(f):
	start_time = time.time()
	f()
	elapsed_time = time.time() - start_time
	print(f"{f.__name__}: {elapsed_time:.3f}s")

def main():
	easy_bench(hnsw_construction)
	easy_bench(hnsw_capped_construction)
	easy_bench(rnn_construction)
	easy_bench(sen_construction)

main()
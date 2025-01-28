use foldhash::HashSet;
use rand::Rng;
use rayon::slice::ParallelSliceMut;

pub fn random_unique_usizes_except(max: usize, n: usize, except: usize) -> Vec<usize> {
	if n >= max {
		/* If n is larger than max, return all numbers from 0 to max-1 */
		(0..max).collect()
	} else if 2*n >= max {
		/* If n is between max/2 and max, get max-n random neighbors instead and return the complement */
		let mut blacklist = HashSet::default();
		blacklist.insert(except);
		random_unique_usizes_except(max, max-n-1, except).into_iter().for_each(|i| _=blacklist.insert(i));
		(0..max).filter(|i| !blacklist.contains(i)).collect()
	} else {
		/* If n is smaller than max/2, create a unique set of n values */
		let mut rng = rand::thread_rng();
		let mut sampled = HashSet::default();
		while sampled.len() < n {
			let sample = rng.gen_range(0..max);
			debug_assert!(sample < max);
			if sample != except {
				sampled.insert(sample);
			}
		}
		sampled.into_iter().collect()
	}
}
pub fn random_usize_pairs(max: usize, n: usize) -> Vec<(usize, usize)> {
	let mut rng = rand::thread_rng();
	(0..n).map(|_| (rng.gen_range(0..max), rng.gen_range(0..max))).collect()
}

pub fn duplicate_free_join<T: Copy+PartialOrd+PartialEq>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
	let mut result: Vec<T> = a.clone();
	result.extend(b.iter().map(|&v|v));
	sort_and_remove_duplicates(&mut result);
	result
}
pub fn par_duplicate_free_join<T: Copy+PartialOrd+PartialEq+Send>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
	let mut result: Vec<T> = a.clone();
	result.extend(b.iter().map(|&v|v));
	par_sort_and_remove_duplicates(&mut result);
	result
}
pub fn par_sort_and_remove_duplicates<T: PartialOrd+PartialEq+Send>(a: &mut Vec<T>) {
	a.par_sort_by(|a, b| unsafe{a.partial_cmp(b).unwrap_unchecked()});
	remove_duplicates(a);
}
pub fn sort_and_remove_duplicates<T: PartialOrd+PartialEq>(a: &mut Vec<T>) {
	a.sort_by(|a, b| unsafe{a.partial_cmp(b).unwrap_unchecked()});
	remove_duplicates(a);
}
pub fn remove_duplicates_with_key<R, T: PartialOrd+PartialEq, K: Fn(&R)->&T>(a: &mut Vec<R>, key: K) {
	if a.len() <= 1 { return; }
	unsafe {
		// Last index of items to keep
		let mut target: usize = 0;
		let mut target_ref = a.get_unchecked(target) as *const R;
		for i in 1..a.len() {
			if key(target_ref.as_ref().unwrap_unchecked()).ne(key(a.get_unchecked(i))) {
				// Move element at i to target (overwriting duplicated between i and target):
				target += 1;
				target_ref = a.get_unchecked(target) as *const R;
				a.swap(target, i);
			}
		}
		a.truncate(target + 1);
	}
}
pub fn remove_duplicates<T: PartialOrd+PartialEq>(a: &mut Vec<T>) {
	remove_duplicates_with_key(a, |x|x)
}
pub fn remove_and_get_duplicates_with_key<R, T: PartialOrd+PartialEq, K: Fn(&R)->&T>(a: &mut Vec<R>, key: K) -> Vec<R> {
	if a.len() <= 1 { vec![] } else { unsafe {
		// Last index of items to keep
		let mut target: usize = 0;
		let mut target_ref = a.get_unchecked(target) as *const R;
		for i in 1..a.len() {
			if key(target_ref.as_ref().unwrap_unchecked()).ne(key(a.get_unchecked(i))) {
				// Move element at i to target (overwriting duplicated between i and target):
				target += 1;
				target_ref = a.get_unchecked(target) as *const R;
				a.swap(target, i);
			}
		}
		a.drain(target + 1..).collect()
	}}
}
pub fn remove_and_get_duplicates<T: PartialOrd+PartialEq>(a: &mut Vec<T>) -> Vec<T> {
	remove_and_get_duplicates_with_key(a, |x|x)
}


#[test]
fn test_random_unique_usizes_except() {
	let max = 1000;
	let except = 5;
	for _ in 0..100 {
		assert_eq!(random_unique_usizes_except(max, 10, except).len(), 10);
		assert_eq!(random_unique_usizes_except(max, max-10, except).len(), max-10);
		assert_eq!(random_unique_usizes_except(max, max/2-1, except).len(), max/2-1);
		assert_eq!(random_unique_usizes_except(max, max/2, except).len(), max/2);
		assert_eq!(random_unique_usizes_except(max, max/2+1, except).len(), max/2+1);
		assert!(random_unique_usizes_except(max, 100, except).into_iter().filter(|&i| i == except).count() == 0);
		assert!(random_unique_usizes_except(max, 100, except).into_iter().max().unwrap() < max);
	}
}

#[test]
fn test_duplicate_free_join() {
	let max = 1500;
	let n = 2000;
	for _ in 0..100 {
		let a: Vec<usize> = (0..n).map(|_| rand::thread_rng().gen_range(0..max)).collect();
		let b: Vec<usize> = (0..n).map(|_| rand::thread_rng().gen_range(0..max)).collect();
		let joined_set = vec![a.clone(),b.clone()].into_iter().flatten().collect::<HashSet<_>>();
		let joined = duplicate_free_join(&a, &b);
		assert_eq!(joined.len(), joined_set.len());
		joined.into_iter().for_each(|i| assert!(joined_set.contains(&i)));
	}
}



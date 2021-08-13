use std::{
    iter::Sum,
    ops::{Mul, MulAssign},
};

use getset::Getters;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, ArrayViewMut2, FixedInitializer,
    RawData,
};
use ndarray_linalg::*;
use ndarray_rand::{rand::thread_rng, rand_distr::WeightedAliasIndex};
use num_traits::Float;
use rand::prelude::*;

#[derive(Debug, Getters, Clone)]
pub struct IsingModel {
    #[get = "pub"]
    J: Array2<f32>,
    #[get = "pub"]
    h: Array1<f32>,
}

trait Model {
    fn init_dim1_spins<R>(N: usize, items: &[(i8, usize); 2], rng: &mut R) -> Array1<i8>
    where
        R: Rng,
    {
        let dist = WeightedAliasIndex::new(items.iter().map(|item| item.1).collect()).unwrap();

        let state = dist
            .sample_iter(rng)
            .take(N)
            .map(|index| items[index].0)
            .collect();

        Array1::from_vec(state)
    }
}

impl From<QuboModel> for IsingModel {
    fn from(item: QuboModel) -> Self {
        let q = item.Q();
        let J = q.clone() + q.t();
        let h = {
            let mut h = Array1::zeros(q.dim().0);
            for k in 0..q.dim().0 {
                h = h + q.column(k).to_owned() + q.row(k);
            }
            h
        };

        IsingModel { J, h }
    }
}

impl IsingModel {
    const CHOICE_ITEMS: [(i8, usize); 2] = [(1, 1), (-1, 1)];
    pub const SPINS_FROM_BITS: fn(&i8) -> i8 = |q| 2 * q - 1;

    pub fn new(J: Array2<f32>, h: Array1<f32>) -> Self {
        let n = J.dim().0;
        IsingModel { J, h }
    }

    pub fn init_trotter_spins<R>(
        N: usize,
        trotter_n: usize,
        is_trotter_copy: bool,
        rng: &mut R,
    ) -> Array2<i8>
    where
        R: Rng,
    {
        let mut init_state = Array2::zeros((trotter_n, N));
        if is_trotter_copy {
            let state = Self::init_spins(N, rng);
            for mut line in init_state.rows_mut() {
                line.assign(&state);
            }
        } else {
            for mut line in init_state.rows_mut() {
                line.assign(&Self::init_spins(N, rng));
            }
        }
        init_state
    }

    pub fn init_spins<R>(N: usize, rng: &mut R) -> Array1<i8>
    where
        R: Rng,
    {
        Self::init_dim1_spins(N, &Self::CHOICE_ITEMS, rng)
    }

    pub fn calculate_energy(&self, spins: ArrayView1<i8>) -> f32 {
        let mut e = 0_f32;
        for i in 0..self.J().dim().0 {
            for j in 0..self.J().dim().1 {
                e += self.J()[[i, j]] * (spins[i] * spins[j]) as f32;
            }
        }

        for i in 0..self.h().dim() {
            e += self.h()[i] * spins[i] as f32;
        }

        e
    }

    pub fn calculate_dE(&self, spins: ArrayView1<i8>, flip_spin: usize) -> f32 {
        let mut dE = self.h()[flip_spin];

        dE += self
            .J()
            .row(flip_spin)
            .indexed_iter()
            .map(|(index, j)| j * spins[index] as f32)
            .sum::<f32>();

        if spins[flip_spin] != 1 {
            dE *= -1.;
        }

        -2. * dE
    }

    pub fn accept_flip(mut spins: ArrayViewMut1<i8>, flip_spin: usize) {
        spins[flip_spin] = -spins[flip_spin];
    }

    pub fn accept_global_flip(mut spins: ArrayViewMut2<i8>, flip_spin: usize) {
        let mut global_spin = spins.column_mut(flip_spin);
        global_spin *= -1;
    }
}

impl Model for IsingModel {}

#[derive(Debug, Getters, Clone)]
pub struct QuboModel {
    #[get = "pub"]
    Q: Array2<f32>,
}

impl QuboModel {
    const CHOICE_ITEMS: [(i8, usize); 2] = [(1, 1), (0, 1)];
    pub const BITS_FROM_SPINS: fn(&i8) -> i8 = |s| (s + 1) / 2;

    pub fn new(Q: Array2<f32>) -> Self {
        let n = Q.dim().0;
        QuboModel { Q }
    }

    pub fn init_trotter_bits<R>(
        N: usize,
        trotter_n: usize,
        is_trotter_copy: bool,
        rng: &mut R,
    ) -> Array2<i8>
    where
        R: Rng,
    {
        IsingModel::init_trotter_spins(N, trotter_n, is_trotter_copy, rng)
            .map(QuboModel::BITS_FROM_SPINS)
    }

    pub fn init_bits<R>(N: usize, rng: &mut R) -> Array1<i8>
    where
        R: Rng,
    {
        Self::init_dim1_spins(N, &Self::CHOICE_ITEMS, rng)
    }

    pub fn calculate_energy(&self, spins: &Array1<i8>) -> f32 {
        let mut e = 0_f32;
        for i in 0..self.Q().dim().0 {
            for j in 0..self.Q().dim().1 {
                e += self.Q()[[i, j]] * (spins[i] * spins[j]) as f32;
            }
        }

        e
    }
}

impl Model for QuboModel {}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::IsingModel;

    #[test]
    fn init_trotter_spins_test() {
        let n = 10;
        let trotter_n = 50;
        let state = IsingModel::init_trotter_spins(n, trotter_n, true, &mut rand::thread_rng());
        let line = state.row(0);

        for l in state.rows() {
            assert_eq!(line, l);
        }

        assert_eq!(trotter_n, state.dim().0);
        assert_eq!(n, state.dim().1);
    }
}

use std::{
    iter::Sum,
    ops::{Add, Mul, MulAssign},
};

use crate::math::DiagonalMatrix;
use getset::Getters;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, ArrayViewMut2, FixedInitializer,
    RawData,
};
use ndarray_linalg::*;
use ndarray_rand::{rand::thread_rng, rand_distr::WeightedAliasIndex};
use num_traits::Float;
use rand::prelude::*;

// NOTE アニーリングマシンに単一スピンを渡してマシン側にコピーさせるべき
pub fn clone_array_row_matrix<T>(array: ArrayView1<T>, p: usize) -> Array2<T>
where
    T: Clone + num_traits::Zero + Default,
{
    let mut matrix = Array2::zeros((p, array.len()));
    for i in 0..p {
        matrix.row_mut(i).assign(&array);
    }

    matrix
}

#[derive(Debug, Getters, Clone)]
pub struct IsingModel {
    #[get = "pub"]
    J: Array2<f64>,
    #[get = "pub"]
    h: Array1<f64>,
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
        // 三角行列化して係数で除算
        let q = (q.clone().into_triangular(UPLO::Upper)
            + (q.clone().into_triangular(UPLO::Lower) - q.to_daigonal_matrix()).t())
            / 4.;

        let J = q.clone().into_triangular(UPLO::Upper) - q.to_daigonal_matrix();
        let h = {
            let mut h = Array1::zeros(q.dim().0);
            for k in 0..q.dim().0 {
                h = h + (q.column(k).to_owned() + q.row(k));
            }
            // 以下と同義
            // for i in 0..q.dim().0 {
            //     h[i] += q.row(i).sum();
            //     h[i] += q.column(i).sum();
            // }

            h
        };

        IsingModel::new(J, h)
    }
}

impl IsingModel {
    const CHOICE_ITEMS: [(i8, usize); 2] = [(1, 1), (-1, 1)];
    pub const SPINS_FROM_BITS: fn(&i8) -> i8 = |q| 2 * q - 1;

    pub fn new(mut J: Array2<f64>, mut h: Array1<f64>) -> Self {
        h = h + J.diag();

        J = &J - J.to_daigonal_matrix();
        J = J.clone().into_triangular(UPLO::Upper) + J.into_triangular(UPLO::Lower).t();

        let j_max = J.iter().fold(0. / 0., |m, v| v.abs().max(f64::abs(m)));
        let h_max = h.iter().fold(0. / 0., |m, v| v.abs().max(f64::abs(m)));
        let max = j_max.abs().max(h_max.abs());

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

    pub fn calculate_energy(&self, spins: ArrayView1<i8>) -> f64 {
        let mut e = 0.;
        for i in 0..self.J().dim().0 {
            for j in 0..self.J().dim().1 {
                if i < j {
                    e += self.J()[[i, j]] * (spins[i] * spins[j]) as f64;
                }
            }
        }

        for i in 0..self.h().len() {
            e += self.h()[i] * spins[i] as f64;
        }

        e
    }

    pub fn calculate_dE(&self, spins: ArrayView1<i8>, flip_spin: usize) -> f64 {
        let mut dE = self.h()[flip_spin];

        for i in 0..spins.dim() {
            dE += self.J[[i, flip_spin]] * spins[i] as f64;
            dE += self.J[[flip_spin, i]] * spins[i] as f64;
        }

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
    Q: Array2<f64>,
}

impl QuboModel {
    const CHOICE_ITEMS: [(i8, usize); 2] = [(1, 1), (0, 1)];
    pub const BITS_FROM_SPINS: fn(&i8) -> i8 = |s| (s + 1) / 2;

    pub fn new(Q: Array2<f64>) -> Self {
        // 三角行列に変換
        let Q = Q.clone().into_triangular(UPLO::Upper)
            + (Q.clone().into_triangular(UPLO::Lower) - Q.to_daigonal_matrix()).t();
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

    pub fn calculate_energy(&self, spins: ArrayView1<i8>) -> f64 {
        let mut e = 0.;
        for i in 0..self.Q().dim().0 {
            for j in 0..self.Q().dim().1 {
                if i <= j {
                    e += self.Q()[[i, j]] * (spins[i] * spins[j]) as f64;
                }
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

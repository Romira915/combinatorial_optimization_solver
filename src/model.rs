use getset::Getters;
use ndarray::{Array, Array1, Array2, FixedInitializer};
use ndarray_linalg::*;
use ndarray_rand::{rand::thread_rng, rand_distr::WeightedAliasIndex};
use num_traits::Float;
use rand::prelude::*;

#[derive(Debug, Getters)]
pub struct IsingModel {
    #[get = "pub"]
    J: Array2<f32>,
    #[get = "pub"]
    h: Array1<f32>,
}

impl From<QuboModel> for IsingModel {
    fn from(item: QuboModel) -> Self {
        let q = item.Q();
        let h = {
            let mut h = Array1::zeros(q.dim().0);
            for k in 0..q.dim().0 {
                h = h + q.column(k).to_owned() + q.row(k);
            }
            h
        };
        IsingModel {
            J: q.clone() + q.t(),
            h,
        }
    }
}

impl IsingModel {
    const ISING_CHOICE_ITEMS: [(i8, usize); 2] = [(1, 1), (-1, 1)];

    pub fn new(J: Array2<f32>, h: Array1<f32>) -> Self {
        IsingModel { J, h }
    }

    pub fn init_trotter_state(N: usize, trotter_n: usize) -> Array2<i8> {
        let dist =
            WeightedAliasIndex::new(Self::ISING_CHOICE_ITEMS.iter().map(|item| item.1).collect())
                .unwrap();
        let mut rng = thread_rng();

        let state = dist
            .sample_iter(&mut rng)
            .take(N * trotter_n)
            .map(|index| Self::ISING_CHOICE_ITEMS[index].0)
            .collect();

        Array2::from_shape_vec((N, trotter_n), state)
            .expect("Failed to init_state. (fn from_shape_vec)")
    }

    pub fn init_state(N: usize) -> Array1<i8> {
        let dist =
            WeightedAliasIndex::new(Self::ISING_CHOICE_ITEMS.iter().map(|item| item.1).collect())
                .unwrap();
        let mut rng = thread_rng();

        let state = dist
            .sample_iter(&mut rng)
            .take(N)
            .map(|index| Self::ISING_CHOICE_ITEMS[index].0);

        Array1::from_iter(state)
    }

    pub fn calculate_energy(&self, spins: &Array1<i8>) -> f32 {
        0.
    }
}

#[derive(Debug, Getters)]
pub struct QuboModel {
    #[get = "pub"]
    Q: Array2<f32>,
}

impl QuboModel {
    pub fn new(Q: Array2<f32>) -> Self {
        QuboModel { Q }
    }
}

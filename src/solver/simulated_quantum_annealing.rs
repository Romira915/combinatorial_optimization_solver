use ndarray::{array, Array, Array1, Array2};
use ndarray_linalg::Scalar;
use num_traits::Float;
use rand::{prelude::StdRng, Rng, SeedableRng};
use std::{f64::consts, sync::Arc};

use crate::{
    model::{IsingModel, QuboModel},
    solver::{self, SolutionRecord, Solver},
};
use getset::Getters;

#[derive(Debug)]
pub struct SimulatedQuantumAnnealing {
    G0: f64,
    Gf: f64,
    T: f64,
    steps: usize,
    N: usize,
    P: usize,
    PT: f64,
    model: Arc<IsingModel>,
    spins: Array2<i8>,
    rng: StdRng,
    record: Option<SolutionRecord>,
}

impl SimulatedQuantumAnnealing {
    const IS_TROTTER_COPY: bool = false;
    pub fn new(
        G0: f64,
        Gf: f64,
        T: f64,
        steps: usize,
        P: usize,
        model: Arc<IsingModel>,
        mut rng: Option<StdRng>,
    ) -> Self {
        let N = model.J().dim().0;
        let mut rng = match rng {
            Some(rng) => rng,
            None => rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
        };
        SimulatedQuantumAnnealing {
            G0,
            Gf,
            T,
            steps,
            N,
            P,
            PT: P as f64 * T,
            model,
            spins: IsingModel::init_trotter_spins(N, P, Self::IS_TROTTER_COPY, &mut rng),
            rng,
            record: None,
        }
    }

    // BUG: この関数はCloneした場合機能しない
    pub fn new_with_spins(
        G0: f64,
        Gf: f64,
        T: f64,
        steps: usize,
        P: usize,
        model: Arc<IsingModel>,
        spins: Array2<i8>,
        mut rng: Option<StdRng>,
    ) -> Self {
        let N = model.J().dim().0;
        let mut rng = match rng {
            Some(rng) => rng,
            None => rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
        };
        println!("{}", spins[[N - 1, N - 1]]);
        SimulatedQuantumAnnealing {
            G0,
            Gf,
            T,
            steps,
            N,
            P,
            PT: P as f64 * T,
            model,
            spins,
            rng,
            record: None,
        }
    }

    fn energy_list(&self) -> Array1<f64> {
        let mut energy_list = Array1::zeros(self.P);
        for (spin, e) in self.spins.rows().into_iter().zip(energy_list.iter_mut()) {
            *e = self.model.calculate_energy(spin);
        }

        energy_list
    }
}

impl Solver for SimulatedQuantumAnnealing {
    fn solve(&mut self) -> crate::solver::SolutionRecord {
        let G_array = Array::linspace(self.G0, self.Gf, self.steps);

        for G in &G_array {
            // local flip
            for k in 0..self.P {
                let flip_local_index = self.rng.gen_range(0..self.N);
                let k = self.rng.gen_range(0..self.P);
                // let J_p = -self.PT
                //     * (G / self.PT).tanh().log(consts::E)
                //     * (self.spins[[k, flip_local_index]]
                //         * (self.spins[[(k + self.P - 1) % self.P, flip_local_index]]
                //             + self.spins[[(k + 1) % self.P, flip_local_index]]))
                //         as f64;

                let B = self.T / 2. * (1.0 / (G / self.PT).tanh()).log(consts::E);
                let delta_trotter = 2.
                    * B
                    * (self.spins[[k, flip_local_index]]
                        * (self.spins[[(k + self.P - 1) % self.P, flip_local_index]]
                            + self.spins[[(k + 1) % self.P, flip_local_index]]))
                        as f64;
                let delta_E = self.model.calculate_dE(self.spins.row(k), flip_local_index) as f64
                    / self.P as f64;

                let delta = delta_E + delta_trotter;
                let (delta_2, delta_2_t) = {
                    let (b, b_q) = {
                        let mut b = 0.;
                        let mut b_q = 0.;
                        for i in 0..self.P {
                            b += self.model.calculate_energy(self.spins.row(i));
                        }
                        b /= self.P as f64;

                        for i in 0..self.N {
                            for k in 0..self.P {
                                b_q += -B
                                    * (self.spins[[k, i]] * self.spins[[(k + 1) % self.P, i]])
                                        as f64;
                            }
                        }

                        (b, b_q)
                    };

                    let (a, a_q) = {
                        let mut a = 0.;
                        let mut a_q = 0.;
                        let mut fix_spins = self.spins.clone();
                        fix_spins[[k, flip_local_index]] = -1 * fix_spins[[k, flip_local_index]];

                        for i in 0..self.P {
                            a += self.model.calculate_energy(fix_spins.row(i));
                        }
                        a /= self.P as f64;

                        for i in 0..self.N {
                            for k in 0..self.P {
                                a_q += -B
                                    * (fix_spins[[k, i]] * fix_spins[[(k + 1) % self.P, i]]) as f64;
                            }
                        }

                        (a, a_q)
                    };

                    (a - b, a_q - b_q)
                };
                // println!("d1 {}", delta_E);
                // println!("d2 {}", delta_2);
                // println!("delta_2_t {}", delta_2_t);
                // println!("delta_trotter {}", delta_trotter);

                let p = f64::min(1., (-(delta_2 + delta_trotter) / self.T).exp());
                if solver::probability_boolean(p, &mut self.rng) {
                    IsingModel::accept_flip(self.spins.row_mut(k), flip_local_index);
                }
            }

            // global flip
            // let flip_global_index = self.rng.gen_range(0..self.N);
            // let delta_E = {
            //     let mut d = 0.;
            //     for k in 0..self.P {
            //         d += self
            //             .model
            //             .calculate_dE(self.spins.row(k), flip_global_index)
            //             as f64;
            //     }
            //     d
            // };
            // let p = f64::min(1., (-delta_E / self.PT).exp());
            // if solver::probability_boolean(p, &mut self.rng) {
            //     IsingModel::accept_global_flip(self.spins.view_mut(), flip_global_index);
            // }
        }

        let energy_list = self.energy_list();
        let (min_index, min_energy) =
            energy_list
                .iter()
                .enumerate()
                .fold((0, f64::NAN), |acc, v| {
                    let v = (v.0, v.1.to_owned());
                    v.min(&acc)
                });

        let record = SolutionRecord {
            solver_name: "Simulated Quantum Annealing".to_string(),
            bits: self.spins.row(min_index).map(QuboModel::BITS_FROM_SPINS),
            energy: min_energy,
            parameter: format!(
                "[G0 {}; Gf {}; steps {}; T {}; P {}; PT {};]",
                self.G0, self.Gf, self.steps, self.T, self.P, self.PT
            ),
        };
        self.record = Some(record.clone());

        // let array = array![-1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1];
        // let opt_e = self.model.calculate_energy(array.view());
        // println!("opt E {}", opt_e);
        // println!("今の {}", self.model.calculate_energy(self.spins.row(0)));

        record
    }

    fn record(&self) -> Option<SolutionRecord> {
        self.record.clone()
    }

    fn clone_solver(&self) -> Self {
        let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();

        let obj = SimulatedQuantumAnnealing {
            G0: self.G0.clone(),
            Gf: self.Gf.clone(),
            T: self.T.clone(),
            steps: self.steps.clone(),
            N: self.N.clone(),
            P: self.P.clone(),
            PT: self.PT.clone(),
            model: self.model.clone(),
            spins: IsingModel::init_trotter_spins(self.N, self.P, Self::IS_TROTTER_COPY, &mut rng),
            rng,
            record: self.record.clone(),
        };

        obj
    }

    fn solver_with_filter(&mut self, filter: &crate::opt::TspNode) -> SolutionRecord {
        let G_array = Array::linspace(self.G0, self.Gf, self.steps);

        for G in &G_array {
            // local flip
            for k in 0..self.P {
                for _ in 0..self.N {
                    let flip_local_index = self.rng.gen_range(0..self.N);
                    let k = self.rng.gen_range(0..self.P);

                    let B = self.T * (1.0 / (G / self.PT).tanh()).log(consts::E);
                    let delta_trotter = B
                        * (self.spins[[k, flip_local_index]]
                            * (self.spins[[(k + self.P - 1) % self.P, flip_local_index]]
                                + self.spins[[(k + 1) % self.P, flip_local_index]]))
                            as f64;
                    let delta_E =
                        self.model.calculate_dE(self.spins.row(k), flip_local_index) as f64;
                    let p = f64::min(1., (-(delta_E + delta_trotter) / self.T).exp());
                    if solver::probability_boolean(p, &mut self.rng) {
                        IsingModel::accept_flip(self.spins.row_mut(k), flip_local_index);
                    }
                }
            }
        }

        let energy_list = self.energy_list();
        let (min_index, min_energy) =
            energy_list
                .iter()
                .enumerate()
                .fold((0, f64::NAN), |acc, v| {
                    let v = (v.0, v.1.to_owned());
                    if acc.1 == f64::NAN {
                        return v;
                    }
                    if acc.1 < v.1 {
                        match (
                            filter.len_from_state(self.spins.row(acc.0).view()),
                            filter.len_from_state(self.spins.row(v.0).view()),
                        ) {
                            (Ok(_), _) => acc,
                            (_, Ok(_)) => v,
                            _ => acc,
                        }
                    } else {
                        match (
                            filter.len_from_state(self.spins.row(acc.0).view()),
                            filter.len_from_state(self.spins.row(v.0).view()),
                        ) {
                            (Ok(_), _) => acc,
                            (_, Ok(_)) => v,
                            _ => v,
                        }
                    }
                });

        let record = SolutionRecord {
            solver_name: "Simulated Quantum Annealing".to_string(),
            bits: self.spins.row(min_index).map(QuboModel::BITS_FROM_SPINS),
            energy: min_energy,
            parameter: format!(
                "[G0 {}; Gf {}; steps {}; T {}; P {}; PT {};]",
                self.G0, self.Gf, self.steps, self.T, self.P, self.PT
            ),
        };
        self.record = Some(record.clone());

        record
    }
}

impl Clone for SimulatedQuantumAnnealing {
    fn clone(&self) -> Self {
        let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();
        SimulatedQuantumAnnealing {
            G0: self.G0.clone(),
            Gf: self.Gf.clone(),
            T: self.T.clone(),
            steps: self.steps.clone(),
            N: self.N.clone(),
            P: self.P.clone(),
            PT: self.PT.clone(),
            model: self.model.clone(),
            spins: IsingModel::init_trotter_spins(self.N, self.P, Self::IS_TROTTER_COPY, &mut rng),
            rng,
            record: self.record.clone(),
        }
    }
}

trait OrdWithTuple {
    fn min(self, other: &Self) -> Self;
}

impl OrdWithTuple for (usize, f64) {
    fn min(self, other: &Self) -> Self {
        let min = self.1.min(other.1);
        if min == self.1 {
            self
        } else {
            *other
        }
    }
}

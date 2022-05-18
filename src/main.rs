use core::num;
use std::env;
use std::process::exit;
use std::sync::Arc;

use combinatorial_optimization_solver::dataset::{load_knapsack, KnapsackData};
use combinatorial_optimization_solver::model::{clone_array_row_matrix, IsingModel, QuboModel};
use combinatorial_optimization_solver::opt::TspNode;
use combinatorial_optimization_solver::solver::annealing_scheduler::AnnealingScheduler;
use combinatorial_optimization_solver::solver::simulated_annealing::SimulatedAnnealing;
use combinatorial_optimization_solver::solver::simulated_quantum_annealing::SimulatedQuantumAnnealing;
use combinatorial_optimization_solver::solver::{Solver, SolverVariant};
use combinatorial_optimization_solver::webhook::Webhook;
use ndarray::{array, s, Array1, Array2, ArrayBase, ArrayView, ArrayView1, Dim, OwnedRepr};
use ndarray_linalg::Scalar;
use num_traits::{pow, Float};
use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use serenity::model::channel::Embed;
use std::convert::TryFrom;
use tokio::time::Instant;

fn number_partitioning(
    n: usize,
    rate: f64,
    rng: &mut StdRng,
) -> (Array1<f64>, f64, Arc<IsingModel>, f64) {
    let numbers = (rng)
        .sample_iter(Uniform::new(0., 1000.))
        .take(n)
        .collect::<Vec<f64>>();
    let sum: f64 = numbers.iter().sum();
    let m = sum * rate;
    let J = {
        let mut J = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                J[[i, j]] = numbers[i] * numbers[j];
            }
        }
        J /= 4.;
        J
    };
    let h = {
        let mut h = Array1::zeros(n);
        let k = sum / 2. - m;
        for i in 0..n {
            h[i] = k * numbers[i];
        }
        h
    };
    let ising = Arc::new(IsingModel::new(J, h));

    let numbers = Array1::from(numbers);
    let constant = (numbers.sum() / 2. - m).powf(2.);

    (numbers, m, ising, constant)
}

fn tsp_ising(rng: &mut StdRng) -> (TspNode, Arc<IsingModel>, f64, f64) {
    let mut tsp = TspNode::try_from("./dataset/ulysses16.tsp").unwrap();
    let max_dist = tsp.max_distance();
    let bias = 0.6;
    tsp.set_bias(max_dist * bias);
    // tsp.set_bias(15.);
    let qubo = QuboModel::from(tsp.clone());
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);

    (tsp, ising, max_dist, bias)
}

fn knapsack(
    n: usize,
    capacity: usize,
    rng: &mut StdRng,
) -> (Arc<IsingModel>, Array1<f64>, Array1<f64>) {
    let cost = rng
        .sample_iter(Uniform::new(1, 10000))
        .take(n)
        .collect::<Array1<u32>>();
    let mut weight = rng
        .sample_iter(Uniform::new(0, (capacity as f64 * 1.3) as u32))
        .take(n)
        .collect::<Array1<u32>>();
    let cost = array![
        94, 506, 416, 992, 649, 237, 457, 815, 446, 422, 791, 359, 667, 598, 7, 544, 334, 766, 994,
        893, 633, 131, 428, 700, 617, 874, 720, 419, 794, 196, 997, 116, 908, 539, 707, 569, 537,
        931, 726, 487, 772, 513, 81, 943, 58, 303, 764, 536, 724, 789,
    ];
    let weight = array![
        485, 326, 248, 421, 322, 795, 43, 845, 955, 252, 9, 901, 122, 94, 738, 574, 715, 882, 367,
        984, 299, 433, 682, 72, 874, 138, 856, 145, 995, 529, 199, 277, 97, 719, 242, 107, 122, 70,
        98, 600, 645, 267, 972, 895, 213, 748, 487, 923, 29, 674,
    ];

    let max_c = cost.iter().max().unwrap().to_owned();

    let Q = {
        let mut Q = Array2::zeros((n, n));
        let B = 40.;
        let A = 10. * B * max_c as f64;
        println!("A: {}", A);
        println!("B: {}", B);

        for i in 0..n {
            for j in 0..n {
                Q[[i, j]] += weight[i] as f64 * weight[i] as f64;

                if i == j {
                    Q[[i, j]] += -2. * A * capacity as f64 * weight[i] as f64;
                    Q[[i, j]] += -B * cost[i] as f64;
                }
            }
        }

        Q
    };

    let qubo = QuboModel::new(Q);
    println!("{:#?}", qubo);
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);
    let cost = cost.map(|i| *i as f64);
    let weight = weight.map(|i| *i as f64);

    (ising, cost, weight)
}

async fn knapsack_log_encode(data: &KnapsackData) -> Arc<IsingModel> {
    let n = data.answer_labels().dim();

    // let cost = array![135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240];
    // let weight = array![70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120,];

    let Q = {
        let max_c = data.costs().iter().max().unwrap().to_owned() as f64;
        let B = 1.;
        let A = max_c * B * 2.;

        println!("A: {}", A);
        println!("B: {}", B);

        let bin_n = (f64::log2((data.capacity() - 1) as f64) + 1.) as usize;
        let mut Q = Array2::zeros((n + bin_n, n + bin_n));

        for i in 0..n {
            for j in 0..n {
                Q[[i, j]] += A * (data.weights()[i] * data.weights()[j]) as f64;

                if i == j {
                    Q[[i, i]] += -B * data.costs()[i] as f64;
                    Q[[i, i]] += A * -2. * (data.capacity() * data.weights()[i]) as f64;
                }
            }
        }

        for i in 0..n {
            for j in n..(n + bin_n) {
                let j_index = j - n;

                Q[[i, j]] += A * 2. * data.weights()[i] as f64 * pow(2., j_index);

                if i == j {}
            }
        }

        for i in n..(n + bin_n) {
            for j in n..(n + bin_n) {
                let i_index = i - n;
                let j_index = j - n;

                Q[[i, j]] += A * pow::<f64>(2., i_index) * pow(2., j_index);

                if i == j {
                    Q[[i, j]] += A * -2. * *data.capacity() as f64 * pow(2., i_index);
                }
            }
        }

        let q_max = Q.iter().fold(0. / 0., |m, v: &f64| v.max(m));
        let q_min = Q.iter().fold(0. / 0., |m, v: &f64| v.min(m));
        println!("Q {}", &Q);
        Q
    };

    let qubo = QuboModel::new(Q);
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);

    ising
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().expect("Not found .env file");
    let url = env::var("WEBHOOK_URL").unwrap();

    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();

    // let (tsp, ising, max_dist, bias) = tsp_ising(&mut rng);
    let datas = load_knapsack("dataset/knapsack/smallcoeff_pisinger/knapPI_1_100_1000.csv")
        .await
        .unwrap();
    let data = &datas[0];
    let ising = knapsack_log_encode(&data).await;
    let y_num = ising.h().len() - data.n();

    let opt_weight_bin = format!("{:b}", data.optimum_solution());
    let opt_weight_bin = {
        let mut vec = Vec::new();
        let mut iter = opt_weight_bin.chars().rev();
        for i in 0..y_num {
            let n = iter.next().unwrap().to_digit(10).unwrap();
            vec.push(n);
        }

        vec
    };

    let steps = 3e4 as usize;
    let try_number_of_times = 30;
    let range_param_start = 5.;
    let range_param_end = 1e-06;
    let T = 0.1;
    let solvers = vec![
        SolverVariant::Sa(SimulatedAnnealing::new(
            range_param_start,
            range_param_end,
            steps,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            T,
            steps,
            1,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            T,
            steps,
            4,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            T,
            steps,
            8,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            T,
            steps,
            32,
            Arc::clone(&ising),
            None,
        )),
    ];

    let scheduler = AnnealingScheduler::new(solvers, try_number_of_times);

    let start = Instant::now();
    let records = scheduler.run();
    let analysis_records = AnnealingScheduler::analysis(&records);

    let end = start.elapsed();

    let embed = Embed::fake(move |e| {
        let mut fields = Vec::new();
        for ar in &analysis_records {
            // let best_len = tsp.len_from_state(ar.best_state.view());
            // let best_len = match best_len {
            //     Ok(len) => len.to_string(),
            //     Err((len, message)) => format!("{} ({})", len, &message),
            // };

            let costs = data.costs().map(|c| *c as f64);
            let weights = data.weights().map(|w| *w as f64);

            let best_spins = Array1::from_iter(ar.best_state.iter().map(|s| s.to_owned() as f64));
            let best_state = best_spins.slice(s![0..*data.n()]);
            let best_bin = best_spins.slice(s![*data.n()..best_spins.len()]);
            let best_cost = best_state.dot(&costs);
            let best_weight = best_state.dot(&weights);
            // let best_hamming = {
            //     let mut d = 0.;
            //     for i in 0..*data.n() {
            //         d += (best_state[i] as f64 - opt[i]).abs();
            //     }
            //     d / n as f64
            // };

            let worst_spins = Array1::from_iter(ar.worst_state.iter().map(|s| s.to_owned() as f64));
            let worst_state = worst_spins.slice(s![0..*data.n()]);
            let worst_bin = worst_spins.slice(s![*data.n()..worst_spins.len()]);
            let worst_cost = worst_state.dot(&costs);
            let worst_weight = worst_state.dot(&weights);
            // let worst_hamming = {
            //     let mut d = 0.;
            //     for i in 0..y_num {
            //         d += (worst_state[i] as f64 - opt[i]).abs();
            //     }
            //     d / n as f64
            // };

            let mut cost_list = Vec::new();
            let mut weight_list = Vec::new();
            for state in &ar.states {
                let state = Array1::from_iter(state.bits.iter().map(|s| s.to_owned() as f64));
                let state = state.slice(s![0..*data.n()]);

                cost_list.push(state.dot(&costs));
                weight_list.push(state.dot(&weights));
            }

            fields.push((
                format!("{}\nparameter {}", ar.solver_name, ar.parameter),
                format!(
                    "[best E {} cost {} weight {}; ave {}; worst E {} cost {} weight {}; \nbits {}\ncost {:?}\nweight {:?}",
                    ar.best_energy,
                    best_cost,
                    best_weight,
                    ar.average_energy,
                    ar.worst_energy,
                    worst_cost,
                    worst_weight,
                    // best_state,
                    "廃止",
                    cost_list,
                    weight_list
                ),
                false,
            ));
            println!("{:?}", &fields);
        }

        e.title("Result")
            .description(format!(
                "try_number_of_times {}; n {}; y_num {} capacity {}; 最適解 {}; 実行時間 {:?}",
                &try_number_of_times,
                &data.n(),
                &y_num,
                &data.capacity(),
                &data.optimum_solution(),
                end
            ))
            .fields(fields)
    });

    let webhook = Webhook::new(&url);
    webhook.send(embed).await.unwrap();

    //     for record in &records {
    //         let cost = record
    //             .bits
    //             .map(|f| *f as f64)
    //             .dot(&Array1::from_iter(numbers.clone().into_iter()));

    //         println!("\n---- result ----");
    //         println!("param [m {}; sum {}]", &m, &sum);
    //         println!("{};\ncost {}", &record, (cost - m).abs());
    //     }
}

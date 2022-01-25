use combinatorial_optimization_solver::opt::TspNode;
use ndarray::Array1;

#[test]
fn tsp_node_test() {
    let node = vec![(3.0, 2.0), (4.0, 5.0), (10.0, 1.0), (8.0, 10.0)];
    let tsp_node = TspNode::new("test data".to_string(), node);
    let state = Array1::from_vec(vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    let len = tsp_node.len_from_state(state.view());

    assert!(len.is_ok())
}

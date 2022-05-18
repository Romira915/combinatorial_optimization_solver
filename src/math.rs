use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use ndarray_linalg::IntoTriangular;

pub fn log_encoder_integer(integer: i32) {
    let num = f64::log2(integer as f64) as i32;
}

pub trait DiagonalMatrix {
    fn to_daigonal_matrix(&self) -> Self;
}

impl<T> DiagonalMatrix for ArrayBase<OwnedRepr<T>, Dim<[usize; 2]>>
where
    T: Clone + num_traits::Zero,
{
    fn to_daigonal_matrix(&self) -> Self {
        let diag = Array2::from_diag(&self.diag().clone());

        diag
    }
}

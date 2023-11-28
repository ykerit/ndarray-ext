use ndarray::{ArrayBase, ArrayD, Axis, Data, Dim, Dimension, IxDynImpl};

pub trait Sorting<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn sort(&self);
    fn argsort(&self);
}

impl<A, S, D> Sorting<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn sort(&self) {
        todo!()
    }

    fn argsort(&self) {
        todo!()
    }
}

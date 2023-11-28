use ndarray::{ArrayBase, ArrayD, Axis, Data, Dim, Dimension, IxDynImpl};

pub trait Indexing<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn indexing(&self);
    fn cond_take(&self, cond: impl Fn(&A) -> bool);
    fn cond_where(&self);
}

impl<A, S, D> Indexing<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn indexing(&self) {
        todo!()
    }

    fn cond_take(&self, cond: impl Fn(&A) -> bool) {
        todo!()
    }

    fn cond_where(&self) {
        todo!()
    }
}

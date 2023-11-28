use ndarray::{arr0, ArrayBase, ArrayD, Axis, Data, Dim, Dimension, IxDynImpl};
use std::cmp::Ordering;

pub trait Searching<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: PartialOrd + Clone + Default,
{
    fn max(&self, axis: Option<Axis>) -> Option<ArrayD<A>>;
    fn min(&self, axis: Option<Axis>) -> Option<ArrayD<A>>;
    fn argmin(&self, axis: Option<Axis>) -> Option<ArrayD<isize>>;
    fn argmax(&self, axis: Option<Axis>) -> Option<ArrayD<isize>>;
}

impl<A, S, D> Searching<A, S, D> for ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: PartialOrd + Clone + Default,
{
    fn max(&self, axis: Option<Axis>) -> Option<ArrayD<A>> {
        arg_func_impl(self, axis, Ordering::Greater).map(|x| x.1)
    }

    fn min(&self, axis: Option<Axis>) -> Option<ArrayD<A>> {
        arg_func_impl(self, axis, Ordering::Less).map(|x| x.1)
    }

    fn argmin(&self, axis: Option<Axis>) -> Option<ArrayD<isize>> {
        arg_func_impl(self, axis, Ordering::Less).map(|x| x.0)
    }

    fn argmax(&self, axis: Option<Axis>) -> Option<ArrayD<isize>> {
        arg_func_impl(self, axis, Ordering::Greater).map(|x| x.0)
    }
}

fn arg_func_impl<A, S, D>(
    src: &ArrayBase<S, D>,
    axis: Option<Axis>,
    ordering: Ordering,
) -> Option<(ArrayD<isize>, ArrayD<A>)>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: PartialOrd + Clone + Default,
{
    if src.is_empty() {
        return None;
    }
    let src = src.view().into_dyn();
    let mut axis = axis;
    if src.ndim() == 1 {
        axis = None
    }
    match axis {
        Some(axis) => {
            assert!(axis.index() < src.ndim());
            let ori_shape = src.shape();
            let shape: Vec<_> = ori_shape
                .iter()
                .enumerate()
                .filter_map(|(x, &y)| if x == axis.index() { None } else { Some(y) })
                .collect();
            let mut res_idx = ArrayD::zeros(shape.clone());
            let mut res = ArrayD::default(shape);
            let dim_size = ori_shape[axis.index()];
            let dims: Vec<_> = (0..dim_size)
                .into_iter()
                .map(|x| src.index_axis(axis, x))
                .collect();
            let mut dims_iter: Vec<_> = dims.iter().map(|x| x.indexed_iter()).collect();
            let mut stop = false;
            loop {
                let mut cur_max: Option<(Dim<IxDynImpl>, &A)> = None;
                let mut cur_max_dim = 0;
                for (dim_i, dim_iter) in dims_iter.iter_mut().enumerate() {
                    if let Some((idx, elem)) = dim_iter.next() {
                        match cur_max {
                            Some((_, val)) => {
                                if elem.partial_cmp(val).unwrap() == ordering {
                                    cur_max = Some((idx, elem));
                                    cur_max_dim = dim_i;
                                }
                            }
                            None => {
                                cur_max = Some((idx, elem));
                            }
                        }
                    } else {
                        stop = true;
                        break;
                    }
                }
                if stop {
                    break;
                }
                let e = res_idx.get_mut(&cur_max.as_ref().unwrap().0).unwrap();
                *e = cur_max_dim as isize;
                let e = res.get_mut(&cur_max.as_ref().unwrap().0).unwrap();
                *e = cur_max.unwrap().1.clone();
            }
            Some((res_idx, res))
        }
        None => {
            let mut cur_max = src.first().unwrap();
            let mut cur_max_idx: Dim<IxDynImpl> = Dim::default();
            for (pattern, elem) in src.indexed_iter() {
                if elem.partial_cmp(cur_max).unwrap() == ordering {
                    cur_max_idx = pattern;
                    cur_max = elem;
                }
            }
            let strides = src.strides();
            let offset: isize = cur_max_idx
                .as_array_view()
                .to_slice()
                .unwrap()
                .iter()
                .zip(strides)
                .map(|(&a, &b)| a as isize * b)
                .sum();
            Some((arr0(offset).into_dyn(), arr0(cur_max.clone()).into_dyn()))
        }
    }
}

#[cfg(test)]
mod tests {

    use ndarray::array;

    use super::*;

    #[test]
    fn test_max() {
        let arr = array![
            [
                [0.84603135, 0.80219274, 0.1883196, 0.65296292],
                [0.29057556, 0.59159524, 0.36616059, 0.08111718],
                [0.79279909, 0.8183319, 0.37232594, 0.05945472],
            ],
            [
                [0.02891179, 0.27566659, 0.82562278, 0.78363779],
                [0.171386, 0.6430765, 0.22884153, 0.08185799],
                [0.04588556, 0.38047977, 0.85814249, 0.72221901],
            ],
        ];
        let max = arr.max(None).unwrap();
        assert_eq!(max, arr0(0.85814249).into_dyn());
        let max = arr.max(Some(Axis(0))).unwrap();
        assert_eq!(
            max,
            array![
                [0.84603135, 0.80219274, 0.82562278, 0.78363779],
                [0.29057556, 0.6430765, 0.36616059, 0.08185799],
                [0.79279909, 0.8183319, 0.85814249, 0.72221901]
            ]
            .into_dyn()
        );
        let max = arr.max(Some(Axis(1))).unwrap();
        assert_eq!(
            max,
            array![
                [0.84603135, 0.8183319, 0.37232594, 0.65296292],
                [0.171386, 0.6430765, 0.85814249, 0.78363779]
            ]
            .into_dyn()
        );
        let max = arr.max(Some(Axis(2))).unwrap();
        assert_eq!(
            max,
            array![
                [0.84603135, 0.59159524, 0.8183319],
                [0.82562278, 0.6430765, 0.85814249]
            ]
            .into_dyn()
        )
    }

    #[test]
    fn test_argmax() {
        let arr = array![
            [
                [0.84603135, 0.80219274, 0.1883196, 0.65296292],
                [0.29057556, 0.59159524, 0.36616059, 0.08111718],
                [0.79279909, 0.8183319, 0.37232594, 0.05945472],
            ],
            [
                [0.02891179, 0.27566659, 0.82562278, 0.78363779],
                [0.171386, 0.6430765, 0.22884153, 0.08185799],
                [0.04588556, 0.38047977, 0.85814249, 0.72221901],
            ],
        ];
        let max = arr.argmax(None).unwrap();
        assert_eq!(max, arr0(22).into_dyn());
        let max = arr.argmax(Some(Axis(0))).unwrap();
        assert_eq!(
            max,
            array![[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1]].into_dyn()
        );
        let max = arr.argmax(Some(Axis(1))).unwrap();
        assert_eq!(max, array![[0, 2, 2, 0], [1, 1, 2, 0]].into_dyn());
        let max = arr.argmax(Some(Axis(2))).unwrap();
        assert_eq!(max, array![[0, 1, 1], [2, 1, 2]].into_dyn())
    }

    #[test]
    fn test_min() {
        let arr = array![
            [
                [0.84603135, 0.80219274, 0.1883196, 0.65296292],
                [0.29057556, 0.59159524, 0.36616059, 0.08111718],
                [0.79279909, 0.8183319, 0.37232594, 0.05945472],
            ],
            [
                [0.02891179, 0.27566659, 0.82562278, 0.78363779],
                [0.171386, 0.6430765, 0.22884153, 0.08185799],
                [0.04588556, 0.38047977, 0.85814249, 0.72221901],
            ],
        ];
        let max = arr.min(None).unwrap();
        assert_eq!(max, arr0(0.02891179).into_dyn());
        let max = arr.min(Some(Axis(0))).unwrap();
        assert_eq!(
            max,
            array![
                [0.02891179, 0.27566659, 0.1883196, 0.65296292],
                [0.171386, 0.59159524, 0.22884153, 0.08111718],
                [0.04588556, 0.38047977, 0.37232594, 0.05945472],
            ]
            .into_dyn()
        );
        let max = arr.min(Some(Axis(1))).unwrap();
        assert_eq!(
            max,
            array![
                [0.29057556, 0.59159524, 0.1883196, 0.05945472],
                [0.02891179, 0.27566659, 0.22884153, 0.08185799],
            ]
            .into_dyn()
        );
        let max = arr.min(Some(Axis(2))).unwrap();
        assert_eq!(
            max,
            array![
                [0.1883196, 0.08111718, 0.05945472],
                [0.02891179, 0.08185799, 0.04588556],
            ]
            .into_dyn()
        )
    }

    #[test]
    fn test_argmin() {
        let arr = array![
            [
                [0.84603135, 0.80219274, 0.1883196, 0.65296292],
                [0.29057556, 0.59159524, 0.36616059, 0.08111718],
                [0.79279909, 0.8183319, 0.37232594, 0.05945472],
            ],
            [
                [0.02891179, 0.27566659, 0.82562278, 0.78363779],
                [0.171386, 0.6430765, 0.22884153, 0.08185799],
                [0.04588556, 0.38047977, 0.85814249, 0.72221901],
            ],
        ];
        let max = arr.argmin(None).unwrap();
        assert_eq!(max, arr0(12).into_dyn());
        let max = arr.argmin(Some(Axis(0))).unwrap();
        assert_eq!(
            max,
            array![[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0],].into_dyn()
        );
        let max = arr.argmin(Some(Axis(1))).unwrap();
        assert_eq!(max, array![[1, 1, 0, 2], [0, 0, 1, 1]].into_dyn());
        let max = arr.argmin(Some(Axis(2))).unwrap();
        assert_eq!(max, array![[2, 3, 3], [0, 3, 0]].into_dyn())
    }
}

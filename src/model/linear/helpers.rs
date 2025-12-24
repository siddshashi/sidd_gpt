use ndarray::Array2;
use rand::rng;
use rand_distr::{Uniform, Normal, Distribution};

pub(super) fn initialize_weights(din: usize, dout: usize, is_ffn: bool) -> 
    Array2<f32> {
    let mut rng = rng();

    if is_ffn {
        let std_dev = (2.0 / din as f32).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();

        Array2::from_shape_fn((dout, din), |_| dist.sample(&mut rng))
    } else {
        let limit = (6.0 / (din as f32 + dout as f32)).sqrt();
        let dist = Uniform::new(-limit, limit).unwrap();

        Array2::from_shape_fn((dout, din), |_| dist.sample(&mut rng))
    }
}

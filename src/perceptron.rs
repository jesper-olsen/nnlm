use std::fmt;

//#[derive(Clone, Copy)]
pub struct Perceptron<const IDIM: usize> {
    pub weights: [f64; IDIM],
}

impl<const IDIM: usize> Default for Perceptron<IDIM> {
    fn default() -> Self {
        Self {
            weights: [0.0; IDIM], // bias included if any
        }
    }
}

impl<const IDIM: usize> fmt::Display for Perceptron<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Perceptron")?;

        write!(f, "Weights: ")?;
        for w in &self.weights {
            write!(f, "{w:>8.2} ")?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<const IDIM: usize> Perceptron<IDIM> {
    pub fn new() -> Self {
        Self {
            weights: [0.0; IDIM], // including bias
        }
    }

    pub fn train(&mut self, data: &[[f64; IDIM]], labels: &[i8]) -> f64 {
        assert_eq!(data.len(), labels.len());

        let mut ee = 0.0;
        for (x, &d) in data.iter().zip(labels.iter()) {
            let y = self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>()
                .signum();
            let error = d as f64 - y;
            for (wj, xj) in self.weights.iter_mut().zip(x.iter()) {
                let lr = 1.0; // Optionally impl annealing, e.g. linear decay
                *wj += lr * error * xj;
            }
            ee += error * error;
        }
        ee /= data.len() as f64;
        ee
    }

    pub fn output(&self, x: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi * xi)
            .sum::<f64>()
    }

    pub fn classify(&self, x: &[f64]) -> i8 {
        self.output(x).signum() as i8
    }
}

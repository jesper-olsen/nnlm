use std::fmt;

#[derive(Copy, Clone)]
pub struct GKernel<const IDIM: usize> {
    pub mean: [f64; IDIM],
    pub var: [f64; IDIM],
}

impl<const IDIM: usize> fmt::Display for GKernel<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GKernel")?;
        write!(f, "mean: ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.mean[j])?;
        }
        writeln!(f)?;
        write!(f, "var:  ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.var[j])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<const IDIM: usize> GKernel<IDIM> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; IDIM],
            var: [1.0; IDIM],
        }
    }

    pub fn reset(&mut self, mean: &[f64; IDIM], var: &[f64; IDIM]) {
        self.mean.copy_from_slice(mean);
        self.var.copy_from_slice(var);
    }

    pub fn split(&mut self, other: &GKernel<IDIM>) {
        self.mean
            .iter_mut()
            .zip(other.mean.iter())
            .for_each(|(m, om)| *m = *om + 0.001);
        self.var.copy_from_slice(&other.var);
    }

    pub fn estimate(&mut self, data: &[[f64; IDIM]]) {
        // Estimate mean and variance - Welford's method
        self.mean.fill(0.0);
        self.var.fill(0.0);
        let mut old_mean = [0.0f64; IDIM];
        for (i, v) in data.iter().enumerate() {
            old_mean.copy_from_slice(&self.mean);
            self.mean
                .iter_mut()
                .zip(v.iter())
                .for_each(|(m, x)| *m += (*x - *m) / (i + 1) as f64);
            for j in 0..IDIM {
                self.var[j] += (v[j] - self.mean[j]) * (v[j] - old_mean[j]);
            }
        }
        self.var
            .iter_mut()
            .for_each(|e| *e /= (data.len() - 1) as f64);
    }

    fn _estimate(&mut self, data: &[[f64; IDIM]]) {
        // Estimate mean and variance - 2 pass
        self.mean.fill(0.0);
        self.var.fill(0.0);

        let n = data.len() as f64;
        for v in data {
            self.mean
                .iter_mut()
                .zip(v.iter())
                .for_each(|(m, x)| *m += *x);
        }
        self.mean.iter_mut().for_each(|m| *m /= n);

        for v in data {
            for i in 0..IDIM {
                self.var[i] += (v[i] - self.mean[i]).powi(2);
            }
        }
        self.var.iter_mut().for_each(|v| *v /= n - 1.0);
    }

    /// Squared Euclidian distance between x and the Gaussian kernel.
    pub fn dist_euc(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| (xi - m).powi(2))
            .sum()
    }

    /// Squared Mahalanobis distance between x and the Gaussian kernel.
    pub fn dist(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .zip(self.var.iter())
            .map(|((&m, &xi), &v)| (xi - m).powi(2) / v)
            .sum()
    }

    /// Log probability of x
    pub fn logp(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        let s: f64 = self
            .var
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).ln())
            .sum();
        -0.5 * (self.dist(x) + s)
    }

    /// Probability of x
    pub fn p(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        let a: f64 = (2.0 * std::f64::consts::PI).powi(IDIM as i32);
        let vproduct = self.var.iter().fold(a, |acc, v| acc * v);
        (1.0 / vproduct.sqrt()) * (-0.5 * self.dist(x)).exp()
    }

    pub fn floor_variances(&mut self, floor: f64) {
        self.var.iter_mut().for_each(|v| *v = v.max(floor));
    }

    pub fn set_variances(&mut self, val: f64) {
        self.var.iter_mut().for_each(|v| *v = val);
    }
}

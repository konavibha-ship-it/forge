// Phase 4: A matrix multiply, written in Rust.
//
// Every layer of every neural network — including the transformer you
// trained in Phase 3 — is fundamentally repeated matrix multiplication.
// PyTorch's speed comes from doing this operation extremely efficiently
// in low-level code. This file is your first step toward writing that
// low-level code yourself.
//
// Rust concepts introduced here (comments mark each one):
//   - structs (grouping data together)
//   - Vec<f32> (a growable list of 32-bit floats — Rust's version of a
//     Python list, but with a fixed, known type)
//   - impl blocks (attaching methods to a struct, like a class's methods)
//   - ownership/borrowing (&self means "borrow, don't take ownership" —
//     Rust's core safety feature, preventing whole categories of bugs
//     that are easy to hit in C/C++)

struct Matrix {
    rows: usize,
    cols: usize,
    // Stored as one flat list, row by row, instead of nested lists —
    // this is how real ML libraries store tensors, since it's much
    // faster than nested structures.
    data: Vec<f32>,
}

impl Matrix {
    // Create a new matrix filled with zeros.
    fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    // Create a matrix from explicit values (row-major order).
    fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert_eq!(data.len(), rows * cols, "data length must match rows * cols");
        Matrix { rows, cols, data }
    }

    // Get the value at (row, col). &self means this borrows the matrix
    // rather than consuming it — we can call this many times without
    // "using up" the matrix.
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    // Set the value at (row, col). &mut self means this borrows the
    // matrix MUTABLY — Rust guarantees at compile time that nothing else
    // can read or write this matrix at the same time, which is what
    // prevents a huge class of bugs common in C/C++.
    fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    // Matrix multiply: self (m x n) * other (n x p) = result (m x p).
    // This is the actual operation — three nested loops, standard
    // textbook matrix multiplication.
    fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "matmul shape mismatch: ({}, {}) x ({}, {})",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }

    // Pretty-print the matrix, for checking output by eye.
    fn print(&self) {
        for i in 0..self.rows {
            let row: Vec<String> = (0..self.cols)
                .map(|j| format!("{:6.2}", self.get(i, j)))
                .collect();
            println!("[{}]", row.join(", "));
        }
    }
}

fn main() {
    // A (2x3) matrix
    let a = Matrix::from_vec(2, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);

    // B (3x2) matrix
    let b = Matrix::from_vec(3, 2, vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ]);

    println!("Matrix A ({}x{}):", a.rows, a.cols);
    a.print();

    println!("\nMatrix B ({}x{}):", b.rows, b.cols);
    b.print();

    // A (2x3) * B (3x2) = result (2x2)
    let result = a.matmul(&b);

    println!("\nA * B ({}x{}):", result.rows, result.cols);
    result.print();

    // Sanity check by hand:
    // result[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // result[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // result[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // result[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    println!("\nExpected: [58.00, 64.00], [139.00, 154.00]");

    assert_eq!(result.get(0, 0), 58.0);
    assert_eq!(result.get(0, 1), 64.0);
    assert_eq!(result.get(1, 0), 139.0);
    assert_eq!(result.get(1, 1), 154.0);

    println!("\nAll checks passed — matrix multiply is correct.");
}
# Macro Einsum for ndarray

## Setup
Install `python` and the `opt_einsum` package.

## Usage
The syntax is similar to `numpy`'s. For each input argument, use `(expr).(axes)`.
Then use a fat arrow `=>`, and for the output axes use `.(axes)` and a semicolon `;`.
After the semicolon, you need to pass the dimensions of each of the axes in order for the
macro to optimize the path at compiletime. See example below

## Example
```rust
let y: Array<f64, _> = einsum!(a.mi, b.nj, c.ijpl, d.op, e.ql => .mnoq; i 2, j 2, l 10, m 40, n 40, o 8, p 7, q 6);
```

---
<p align="center">
    <img style="max-height: 10rem" src="https://raw.githubusercontent.com/Case-Rocket-Team/SirinBranding/main/sirin-banner.png">
</p>
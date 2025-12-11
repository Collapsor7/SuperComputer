// -----------------------------
// 3D Wave equation
// Variant 2
// Author:  shihui
// -----------------------------

#include <stdio.h>
#include <cmath>
#include <utility>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

using namespace std;
const double PI = 3.141592653589793238462643383279502884;

static size_t idx(int i, int j, int k, int Ny, int Nz){
    return (static_cast<size_t>(i) * Ny + j) * Nz + k;
}            // {


int main(int argc, char **argv)
{
  int Nx, Ny, Nz, k;
  double Lx, Ly, Lz, T;
  int N = stoi(argv[1]);
  Nx = N, Ny = N, Nz = N;
  int L = 1;
  Lx = L, Ly = L, Lz = L;
  T = 0.01;
  k = stoi(argv[2]);
  int n_thread = stoi(argv[3]);

  const double hx = Lx / (Nx-1);
  const double hy = Ly / (Ny-1);
  const double hz = Lz / Nz;
  const double tau = T / k;

  const double a2 = 1.0 / (PI * PI);
  const double a = sqrt(a2);
  const double at = sqrt((1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz)));
  const double cfl2 = a2 * (tau * tau) * ((1.0 / (hx * hx)) + (1.0 / (hy * hy)) + (1.0 / (hz * hz)));

  cout << "Nx Ny Nz = " << Nx << ' ' << Ny << ' ' << Nz << endl;
  cout << "Lx Ly Lz = " << Lx << ' ' << Ly << ' ' << Lz << "\n";
  cout << "T K = " << T << ' ' << k << ", tau=" << tau << endl;
  cout << "hx hy hz = " << hx << ' ' << hy << ' ' << hz << endl;
  cout << "a^2=" << a2 << ", a=" << a << ", a_t=" << at << endl;
  cout << "CFL^2 = " << cfl2 << (cfl2<=1.0 ? " (OK)" : " (NO!)") << endl;;

  omp_set_num_threads(n_thread);

  const size_t Nnodes = Nx * Ny * Nz;
  double* u_prev = new double[Nnodes];
  double* u_curr = new double[Nnodes];
  double* u_next = new double[Nnodes];

  // #pragma omp parallel for collapse(3) schedule(static)
  // for (int i = 0; i < NX; ++i)
  //   for (int j = 0; j < NY; ++j)
  //     for (int k = 0; k < NZ; ++k) {
  //       const size_t id = idx(i, j, k, NY, NZ);
  //       u_prev[id] = 0.0;
  //       u_curr[id] = 0.0;
  //       u_next[id] = 0.0;
  //     }

  auto askUanalytical = [&](int i, int j, int k, double t) -> double
  {
    const double x = i * hx;
    const double y = j * hy;
    const double z = k * hz;
    return sin(PI * x / Lx) * sin(PI * y / Ly) * sin(2 * PI * z / Lz) * cos(at * t + 2 * PI);
  };

  //init u0
  #pragma omp parallel for collapse(3) schedule(static)
  for(int i = 0; i < Nx; i++){
    for (int j = 0; j < Ny; j++){
        for (int k = 0; k < Nz; k++){
          u_prev[idx(i, j, k, Ny, Nz)] = askUanalytical(i, j, k, 0);
        }
    }
  }
  //Init u1 = u0 + 0.5 * a^2 * tau^2 * (Δ_h phi)
  #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 0; i < Nx;i++){
    for (int j = 0; j < Ny;j++){
      for (int k = 0; k < Nz;k++){
        const size_t id = idx(i, j, k, Ny, Nz);
        if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1){
          u_curr[id] = 0;
        }else{
          int k1, k2;
          k1 = (k == 0 ? Nz-1 : k - 1);
          k2 = (k == Nz-1 ? 0 : k + 1);
          const double ijk = askUanalytical(i, j, k, 0);
          const double i1jk = askUanalytical(i - 1, j, k, 0);
          const double i2jk = askUanalytical(i + 1, j, k, 0);
          const double ij1k = askUanalytical(i, j - 1, k, 0);
          const double ij2k = askUanalytical(i, j + 1, k, 0);
          const double ijk1 = askUanalytical(i, j, k1, 0);
          const double ijk2 = askUanalytical(i, j, k2, 0);
          const double phi = ((i1jk - 2 * ijk + i2jk) / (hx * hx) + (ij1k - 2 * ijk + ij2k) / (hy * hy) + (ijk1 - 2 * ijk + ijk2) / (hz * hz));
          u_curr[id] = u_prev[id] + 0.5 * a2 * tau * tau * phi;
        }
      }
    }
  }

  // mian_iter
  const auto t0 = chrono::high_resolution_clock::now();
  for (int t = 1; t < k;t++){
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Nx;i++){
      for (int j = 0; j < Ny;j++){
        for (int k = 0; k < Nz;k++){
          const size_t id = idx(i, j, k, Ny, Nz);
          if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1){
            u_next[id] = 0;
          }else{
            int k1, k2;
            k1 = (k == 0 ? Nz-1 : k -1);
            k2 = (k == Nz-1 ? 0 : k + 1);
            const double ijk = u_curr[id];
            const double i1jk = u_curr[idx(i - 1, j, k, Ny, Nz)];
            const double i2jk = u_curr[idx(i + 1, j, k, Ny, Nz)];
            const double ij1k = u_curr[idx(i, j - 1, k, Ny, Nz)];
            const double ij2k = u_curr[idx(i, j + 1, k, Ny, Nz)];
            const double ijk1 = u_curr[idx(i, j, k1, Ny, Nz)];
            const double ijk2 = u_curr[idx(i, j, k2, Ny, Nz)];
            const double phi = ((i1jk - 2 * ijk + i2jk) / (hx * hx) + (ij1k - 2 * ijk + ij2k) / (hy * hy) + (ijk1 - 2 * ijk + ijk2) / (hz * hz));
            u_next[id] = 2*u_curr[id] - u_prev[id] + a2 * tau * tau * phi;
          }
        }
      }
    }
    swap(u_prev, u_curr);
    swap(u_curr, u_next);
  }

  const auto t1 = chrono::high_resolution_clock::now();
  const double elapsed = chrono::duration<double>(t1 - t0).count();

  cout << elapsed << endl;

  double max_abs_err = 0.0;
  #pragma omp parallel for reduction(max:max_abs_err) schedule(static)
  for (int i = 0; i < Nx;i++){
    for (int j = 0; j < Ny;j++){
      for (int k = 0; k < Nz;k++){
        const double uref = askUanalytical(i, j, k, T);
        const double err = fabs(u_curr[idx(i, j, k, Ny, Nz)] - uref);
        if (err > max_abs_err) max_abs_err = err;
      }
    }
  }
  cout << "[Result] Time(s)=" << elapsed << ", MaxAbsError=" << scientific << max_abs_err << endl;

  delete[] u_prev;
  delete[] u_curr;
  delete[] u_next;
  return 0;
}
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
#include <mpi.h>

using namespace std;
const double PI = 3.141592653589793238462643383279502884;

static size_t idx(int i, int j, int k, int Ny, int Nz)
{
  return (static_cast<size_t>(i) * Ny + j) * Nz + k;
}

void askLocalStart(int N, int dims, int coord, int &start, int &localN)
{
  int base = N / dims;
  int left = N % dims;

  if (coord < left)
  {
    localN = base + 1;
    start = coord * localN;
  }
  else
  {
    localN = base;
    start = left * (base + 1) + (coord - left) * base;
  }
}
int main(int argc, char **argv)
{

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 3)
  {
    if (rank == 0)
    {
      cerr << "Usage: " << argv[0] << " N K n_threads\n"
           << "  N         : grid size in each dimension (Nx=Ny=Nz=N)\n"
           << "  K         : number of time steps\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int Nx, Ny, Nz, K;
  double Lx, Ly, Lz, T;
  int N = stoi(argv[1]);
  Nx = N, Ny = N, Nz = N;
  int L = 1;
  Lx = L, Ly = L, Lz = L;
  T = 0.01;
  K = stoi(argv[2]);
  // int n_thread = stoi(argv[3]);

  const double hx = Lx / (Nx - 1);
  const double hy = Ly / (Ny - 1);
  const double hz = Lz / Nz;
  const double tau = T / K;

  const double a2 = 1.0 / (PI * PI);
  const double a = sqrt(a2);
  const double at = sqrt((1.0 / (Lx * Lx) + 1.0 / (Ly * Ly) + 4.0 / (Lz * Lz)));
  const double cfl2 = a2 * (tau * tau) * ((1.0 / (hx * hx)) + (1.0 / (hy * hy)) + (1.0 / (hz * hz)));
  const double left = a2 * tau * tau;
  const double inv_hx2 = 1.0 /(hx * hx);
  const double inv_hy2 = 1.0 /(hy * hy);
  const double inv_hz2 = 1.0 /(hz * hz);

  if (rank == 0)
  {
    cout << "Nx Ny Nz = " << Nx << ' ' << Ny << ' ' << Nz << endl;
    cout << "Lx Ly Lz = " << Lx << ' ' << Ly << ' ' << Lz << "\n";
    cout << "T K = " << T << ' ' << K << ", tau=" << tau << endl;
    cout << "hx hy hz = " << hx << ' ' << hy << ' ' << hz << endl;
    cout << "a^2=" << a2 << ", a=" << a << ", a_t=" << at << endl;
    cout << "CFL^2 = " << cfl2 << (cfl2 <= 1.0 ? " (OK)" : " (NO!)") << endl;
    ;
  }

  MPI_Comm cart_comm;
  int dims[3] = {0, 0, 0};
  int periods[3] = {0, 0, 1};
  int coords[3];
  int i_start, j_start, k_start;
  int Nx_local, Ny_local, Nz_local;
  int nbr_x1, nbr_x2;
  int nbr_y1, nbr_y2;
  int nbr_z1, nbr_z2;

  // int dims[3] = {0, 0, 0};
  MPI_Dims_create(size, 3, dims);

  // int periods[3] = {0, 0, 1};
  // MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

  // int coords[3];
  MPI_Cart_coords(cart_comm, rank, 3, coords);
  int px = coords[0];
  int py = coords[1];
  int pz = coords[2];

  if (rank == 0)
  {
    cout << "Process grid Px=" << dims[0]
         << ", Py=" << dims[1]
         << ", Pz=" << dims[2] << "\n";
  }

  // int i_start, j_start, k_start;
  // int Nx_local, Ny_local, Nz_local;

  askLocalStart(Nx, dims[0], coords[0], i_start, Nx_local);
  askLocalStart(Ny, dims[1], coords[1], j_start, Ny_local);
  askLocalStart(Nz, dims[2], coords[2], k_start, Nz_local);

  // if(rank==0)
  // {

  //   cout << "local_size on p(rank = " << rank << ") Nx_local = " << Nx_local
  //        << ", Ny_local = " << Ny_local
  //        << ", Nz_local = " << Nz_local << endl;
  // }

  // find_nbrs
  // int nbr_x1, nbr_x2;
  // int nbr_y1, nbr_y2;
  // int nbr_z1, nbr_z2;

  MPI_Cart_shift(cart_comm, 0, 1, &nbr_x1, &nbr_x2);
  MPI_Cart_shift(cart_comm, 1, 1, &nbr_y1, &nbr_y2);
  MPI_Cart_shift(cart_comm, 2, 1, &nbr_z1, &nbr_z2);

  size_t Nnodes;
  Nnodes = (Nx_local + 2) * (Ny_local + 2) * (Nz_local + 2);

  double *u_prev = new double[Nnodes];
  double *u_curr = new double[Nnodes];
  double *u_next = new double[Nnodes];

  auto askUanalytical = [&](int i, int j, int k, double t) -> double
  {
    const double x = i * hx;
    const double y = j * hy;
    const double z = k * hz;
    return sin(PI * x / Lx) * sin(PI * y / Ly) * sin(2 * PI * z / Lz) * cos(at * t + 2 * PI);
  };


  // ------------------------------------------------------------------------------------

  // init u0

  // #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i <= Nx_local; ++i)
  {
    for (int j = 1; j <= Ny_local; ++j)
    {
      for (int k = 1; k <= Nz_local; ++k)
      {

        int i_global = i_start + (i - 1);
        int j_global = j_start + (j - 1);
        int k_global = k_start + (k - 1);

        u_prev[idx(i, j, k, Ny_local + 2, Nz_local + 2)] = askUanalytical(i_global, j_global, k_global, 0.0);
      }
    }
  }
  // ------------------------------------------------------------------------------------

  // Init u1 = u0 + 0.5 * a^2 * tau^2 * (Δ_h phi)
  // #pragma omp parallel for collapse(3) schedule(static)
  for (int i = 1; i <= Nx_local; i++)
  {
    for (int j = 1; j <= Ny_local; j++)
    {
      for (int k = 1; k <= Nz_local; k++)
      {

        int i_global = i_start + i - 1;
        int j_global = j_start + j - 1;
        int k_global = k_start + k - 1;

        const size_t id = idx(i, j, k, Ny_local + 2, Nz_local + 2);
        if (i_global == 0 || i_global == Nx - 1 || j_global == 0 || j_global == Ny - 1)
        {
          u_curr[id] = 0;
        }
        else
        {
          int k1, k2;
          k1 = (k_global == 0 ? Nz - 1 : k_global - 1);
          k2 = (k_global == Nz - 1 ? 0 : k_global + 1);
          const double ijk = askUanalytical(i_global, j_global, k_global, 0);
          const double i1jk = askUanalytical(i_global - 1, j_global, k_global, 0);
          const double i2jk = askUanalytical(i_global + 1, j_global, k_global, 0);
          const double ij1k = askUanalytical(i_global, j_global - 1, k_global, 0);
          const double ij2k = askUanalytical(i_global, j_global + 1, k_global, 0);
          const double ijk1 = askUanalytical(i_global, j_global, k1, 0);
          const double ijk2 = askUanalytical(i_global, j_global, k2, 0);
          const double phi = ((i1jk - 2 * ijk + i2jk) * inv_hx2 + (ij1k - 2 * ijk + ij2k) * inv_hy2 + (ijk1 - 2 * ijk + ijk2) * inv_hz2);
          u_curr[id] = u_prev[id] + 0.5 * left * phi;
        }
      }
    }
  }

  MPI_Datatype xy_plane;
  {
    int count = (Nx_local + 2) * (Ny_local + 2);
    int blocklen = 1;
    int stride = Nz_local + 2;
    MPI_Type_vector(count, blocklen, stride, MPI_DOUBLE, &xy_plane);
    MPI_Type_commit(&xy_plane);
  }

  MPI_Datatype xz_plane;
  {
    int count = Nx_local + 2;
    int blocklen = Nz_local + 2;
    int stride = (Ny_local + 2) * (Nz_local + 2);
    MPI_Type_vector(count, blocklen, stride, MPI_DOUBLE, &xz_plane);
    MPI_Type_commit(&xz_plane);
  }

  MPI_Datatype yz_plane;
  {
    int count = 1;
    int blocklen = (Ny_local + 2) * (Nz_local + 2);
    int stride = 0;
    MPI_Type_vector(count, blocklen, stride, MPI_DOUBLE, &yz_plane);
    MPI_Type_commit(&yz_plane);
  }
  // int yz_plane_size = (Ny_local + 2) * (Nz_local + 2);

  // ------------------------------------------------------------------------------------

  // mian_iter

  // MPI_Barrier(cart_comm);
  double elapsed_time;
  double t0 = MPI_Wtime();

  const size_t stride_x = (Ny_local + 2) * (Nz_local + 2);
  const size_t stride_y = Nz_local + 2;

  auto update_point = [&](int i, int j, int k) {

    int i_global = i_start + i - 1;
    int j_global = j_start + j - 1;
    int k_global = k_start + k - 1;

    const size_t id   = idx(i, j, k, Ny_local + 2, Nz_local + 2);

    if (i_global == 0 || i_global == Nx - 1 ||
        j_global == 0 || j_global == Ny - 1) {
        u_next[id] = 0.0;
        return;
    }

    size_t base    = id;
    const double ijk  = u_curr[base];
    const double i1jk = u_curr[base - stride_x];
    const double i2jk = u_curr[base + stride_x];
    const double ij1k = u_curr[base - stride_y];
    const double ij2k = u_curr[base + stride_y];
    const double ijk1 = u_curr[base - 1];
    const double ijk2 = u_curr[base + 1];

    const double phi =
        (i1jk - 2.0 * ijk + i2jk) * inv_hx2 +
        (ij1k - 2.0 * ijk + ij2k) * inv_hy2 +
        (ijk1 - 2.0 * ijk + ijk2) * inv_hz2;

    u_next[id] = 2.0 * u_curr[id] - u_prev[id] + left * phi;
  };


  for (int t = 1; t < K; t++)
  {

    MPI_Request reqs[12];
    int r = 0;

    if (nbr_x1 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(0, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, yz_plane, nbr_x1, 102, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(1, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, yz_plane, nbr_x1, 101, cart_comm, &reqs[r++]);
    }

    if (nbr_x2 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(Nx_local + 1, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, yz_plane, nbr_x2, 101, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(Nx_local, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, yz_plane, nbr_x2, 102, cart_comm, &reqs[r++]);
    }

    if (nbr_y1 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(0, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, xz_plane, nbr_y1, 202, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(0, 1, 0, Ny_local + 2, Nz_local + 2)],
                1, xz_plane, nbr_y1, 201, cart_comm, &reqs[r++]);
    }

    if (nbr_y2 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(0, Ny_local + 1, 0, Ny_local + 2, Nz_local + 2)],
                1, xz_plane, nbr_y2, 201, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(0, Ny_local, 0, Ny_local + 2, Nz_local + 2)],
                1, xz_plane, nbr_y2, 202, cart_comm, &reqs[r++]);
    }

    // if (nbr_z1 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(0, 0, 0, Ny_local + 2, Nz_local + 2)],
                1, xy_plane, nbr_z1, 302, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(0, 0, 1, Ny_local + 2, Nz_local + 2)],
                1, xy_plane, nbr_z1, 301, cart_comm, &reqs[r++]);

    }

    // if (nbr_z2 != MPI_PROC_NULL)
    {
      MPI_Irecv(&u_curr[idx(0, 0, Nz_local + 1, Ny_local + 2, Nz_local + 2)],
                1, xy_plane, nbr_z2, 301, cart_comm, &reqs[r++]);
      MPI_Isend(&u_curr[idx(0, 0, Nz_local, Ny_local + 2, Nz_local + 2)],
                1, xy_plane, nbr_z2, 302, cart_comm, &reqs[r++]);
    }

    // MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);

    // #pragma omp parallel for schedule(static)
    for (int i = 2; i <= Nx_local-1; i++)
    {
      for (int j = 2; j <= Ny_local-1; j++)
      {
        for (int k = 2; k <= Nz_local-1; k++)
        {
          int i_global = i_start + i - 1;
          int j_global = j_start + j - 1;
          int k_global = k_start + k - 1;

          const size_t id = idx(i, j, k, Ny_local + 2, Nz_local + 2);
          // if (i_global == 0 || i_global == Nx - 1 || j_global == 0 || j_global == Ny - 1)
          // {
          //   u_next[id] = 0;
          // }
          // else
          // {
            // size_t stride_x = (Ny_local + 2) * (Nz_local + 2);
            // size_t stride_y = Nz_local + 2;

          size_t base = id;
          // const double ijk = u_curr[id];
          // const double i1jk = u_curr[idx(i - 1, j, k, Ny_local + 2, Nz_local + 2)];
          // const double i2jk = u_curr[idx(i + 1, j, k, Ny_local + 2, Nz_local + 2)];
          // const double ij1k = u_curr[idx(i, j - 1, k, Ny_local + 2, Nz_local + 2)];
          // const double ij2k = u_curr[idx(i, j + 1, k, Ny_local + 2, Nz_local + 2)];
          // const double ijk1 = u_curr[idx(i, j, k - 1, Ny_local + 2, Nz_local + 2)];
          // const double ijk2 = u_curr[idx(it, j, k + 1, Ny_local + 2, Nz_local + 2)];
          const double ijk = u_curr[base];
          const double i1jk = u_curr[base - stride_x];
          const double i2jk = u_curr[base + stride_x];
          const double ij1k = u_curr[base - stride_y];
          const double ij2k = u_curr[base + stride_y];
          const double ijk1 = u_curr[base - 1];
          const double ijk2 = u_curr[base + 1];
          const double phi = ((i1jk - 2 * ijk + i2jk) * inv_hx2 + (ij1k - 2 * ijk + ij2k) * inv_hy2 + (ijk1 - 2 * ijk + ijk2) * inv_hz2);
          u_next[id] = 2 * u_curr[id] - u_prev[id] + left * phi;
          // }
        }
      }
    }

    MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);

    //x
    for (int j = 1; j <= Ny_local; ++j)
      for (int k = 1; k <= Nz_local; ++k) {
        update_point(1, j, k);
        update_point(Nx_local, j, k);
      }

    //y
    for (int i = 2; i <= Nx_local-1; ++i)
      for (int k = 1; k <= Nz_local; ++k) {
        update_point(i, 1, k);
        update_point(i, Ny_local, k);
      }

    //z
    for (int i = 2; i <= Nx_local-1; ++i)
      for (int j = 2; j <= Ny_local-1; ++j) {
        update_point(i, j, 1);
        update_point(i, j, Nz_local);
    }
  
    swap(u_prev, u_curr);
    swap(u_curr, u_next);
  }

  double t1 = MPI_Wtime();
  double elapsed_local = t1 - t0;

  MPI_Reduce(&elapsed_local, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

  // cout << elapsed_time << endl;

  double max_abs_err = 0.0;

  double max_abs_err_local = 0.0;
  // #pragma omp parallel for reduction(max : max_abs_err_local) schedule(static)
  for (int i = 1; i <= Nx_local; i++)
  {
    for (int j = 1; j <= Ny_local; j++)
    {
      for (int k = 1; k <= Nz_local; k++)
      {

        int i_global = i_start + i - 1;
        int j_global = j_start + j - 1;
        int k_global = k_start + k - 1;

        const double uref = askUanalytical(i_global, j_global, k_global, T);
        const double err = fabs(u_curr[idx(i, j, k, Ny_local + 2, Nz_local + 2)] - uref);
        if (err > max_abs_err_local)
          max_abs_err_local = err;
      }
    }
  }
  MPI_Allreduce(&max_abs_err_local, &max_abs_err, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

  if (rank == 0)
  {
    cout << "[Result] Time(s)=" << elapsed_time << ", MaxAbsError=" << scientific << max_abs_err << endl;
  }

  MPI_Type_free(&xy_plane);
  MPI_Type_free(&xz_plane);
  MPI_Type_free(&yz_plane);
  MPI_Comm_free(&cart_comm);

  delete[] u_prev;
  delete[] u_curr;
  delete[] u_next;

  MPI_Finalize();
  return 0;
}
#include "voro.h"
#include "schmeud/src/nlist/voro.rs.h"

namespace voro {

std::unique_ptr<container_periodic>
new_container_periodic(double bx_, double bxy_, double by_, double bxz_,
                       double byz_, double bz_, int nx_, int ny_, int nz_,
                       int init_mem_) {
  auto con = std::make_unique<container_periodic>(
      bx_, bxy_, by_, bxz_, byz_, bz_, nx_, ny_, nz_, init_mem_);

  return con;
}

std::unique_ptr<voronoicell_neighbor> new_voronoicell_neighbor() {
  auto v = std::make_unique<voronoicell_neighbor>();
  return v;
}

std::unique_ptr<c_loop_all_periodic>
new_c_loop_all_periodic(const container_periodic &con_) {
  // auto con = const_cast<container_periodic &>(con_);
  auto cl = std::make_unique<c_loop_all_periodic>(con_);
  return cl;
}

std::unique_ptr<std::vector<int>> new_i32_vector() {
  auto v = std::make_unique<std::vector<int>>();
  return v;
}

std::unique_ptr<std::vector<double>> new_f64_vector() {
  auto v = std::make_unique<std::vector<double>>();
  return v;
}

void compute_cell(voronoicell_neighbor &cell_, const container_periodic &con_,
                  const c_loop_all_periodic &voro_loop_) {
  // SAFETY: we can safely perform this conversion because "compute_cell" does
  // not mutate the container or the c_loop, even though the signature implies
  // that it does
  container_periodic &con = const_cast<container_periodic &>(con_);
  c_loop_all_periodic &voro_loop = const_cast<c_loop_all_periodic &>(voro_loop_);
  con.compute_cell(cell_, voro_loop);
}

} // namespace voro
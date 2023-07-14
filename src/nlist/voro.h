#pragma once
// #include "../../extern/voro/src/voro++.hh"
#include "voro++/voro++.hh"
#include <memory>

namespace voro {

std::unique_ptr<container_periodic>
new_container_periodic(double bx_, double bxy_, double by_, double bxz_,
                       double byz_, double bz_, int nx_, int ny_, int nz_,
                       int init_mem_);

std::unique_ptr<voronoicell_neighbor> new_voronoicell_neighbor();

std::unique_ptr<c_loop_all_periodic>
new_c_loop_all_periodic(const container_periodic &con_);

std::unique_ptr<std::vector<int>> new_i32_vector();

std::unique_ptr<std::vector<double>> new_f64_vector();

void compute_cell(voronoicell_neighbor &cell, const container_periodic &con,
                  const c_loop_all_periodic &voro_loop);

} // namespace voro
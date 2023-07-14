#include "voro++/voro++.hh"
#include <random>

using namespace voro;

// Set up constants for the container geometry
const double x_min = -5, x_max = 5;
const double y_min = -5, y_max = 5;
const double z_min = 0, z_max = 10;

// Set up the number of blocks that the container is divided into
const int n_x = 3, n_y = 3, n_z = 3;

int main() {

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-5.0, 5.0);

  container_periodic;
  voro::optimal_particles;
  voro::voronoicell_neighbor;
  voro::c_loop_all_periodic;

  container container(x_min, x_max, y_min, y_max, z_min, z_max, n_x, n_y, n_z,
                      true, true, true, 3);

  // Randomly add particles into the container
  for (int i = 0; i < 100; i++) {
    container.put(i, distribution(generator), distribution(generator),
                  distribution(generator) + 5.0);
  }

  // Save the Voronoi network of all the particles to text files
  // in gnuplot and POV-Ray formats
  //   container.draw_cells_gnuplot("pack_ten_cube.gnu");
  //   container.draw_cells_pov("pack_ten_cube_v.pov");

  // Output the particles in POV-Ray format
  //   container.draw_particles_pov("pack_ten_cube_p.pov");
}
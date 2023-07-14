#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_3.h>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_2<K> Triangulation;
typedef Triangulation::Vertex_circulator Vertex_circulator;
typedef Triangulation::Point Point;
typedef CGAL::Triangulation_3<K> Triangulation3;

int main() {
  std::ifstream in("test.cin");
  std::istream_iterator<Point> begin(in);
  std::istream_iterator<Point> end;
  Triangulation t;
  t.insert(begin, end);
  Vertex_circulator vc = t.incident_vertices(t.infinite_vertex()), done(vc);
  if (vc != nullptr) {
    do {
      std::cout << vc->point() << std::endl;
    } while (++vc != done);
  }
  return 0;
}
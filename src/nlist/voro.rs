use crate::boxdim::BoxDim;
use crate::nlist;

use glam::{DVec3, Vec3};
use numpy::*;
use pyo3::prelude::*;

#[cxx::bridge(namespace = "voro")]
mod ffi {

    unsafe extern "C++" {
        include!("schmeud/src/nlist/voro.h");

        type container_periodic;
        type voronoicell_neighbor;
        type c_loop_all_periodic;

        fn new_container_periodic(
            bx: f64,
            bxy: f64,
            by: f64,
            bxz: f64,
            byz: f64,
            bz: f64,
            nx: i32,
            ny: i32,
            nz: i32,
            init_mem: i32,
        ) -> UniquePtr<container_periodic>;
        fn put(self: Pin<&mut container_periodic>, n: i32, x: f64, y: f64, z: f64);
        fn compute_cell(
            self: Pin<&mut container_periodic>,
            cell: Pin<&mut voronoicell_neighbor>,
            c_loop: Pin<&c_loop_all_periodic>,
        ) -> bool;

        fn new_voronoicell_neighbor() -> UniquePtr<voronoicell_neighbor>;
        fn face_areas(self: Pin<&mut voronoicell_neighbor>, v: Pin<&mut CxxVector<f64>>);
        fn face_vertices(self: Pin<&mut voronoicell_neighbor>, v: Pin<&mut CxxVector<i32>>);
        fn neighbors(self: Pin<&mut voronoicell_neighbor>, v: Pin<&mut CxxVector<i32>>);
        fn normals(self: Pin<&mut voronoicell_neighbor>, v: Pin<&mut CxxVector<f64>>);
        fn vertices(
            self: Pin<&mut voronoicell_neighbor>,
            x: f64,
            y: f64,
            z: f64,
            v: Pin<&mut CxxVector<f64>>,
        );
        fn volume(self: Pin<&mut voronoicell_neighbor>) -> f64;

        fn new_c_loop_all_periodic(
            container: Pin<&container_periodic>,
        ) -> UniquePtr<c_loop_all_periodic>;
        fn start(self: Pin<&mut c_loop_all_periodic>) -> bool;
        fn pid(self: Pin<&mut c_loop_all_periodic>) -> i32;
        fn x(self: Pin<&mut c_loop_all_periodic>) -> f64;
        fn y(self: Pin<&mut c_loop_all_periodic>) -> f64;
        fn z(self: Pin<&mut c_loop_all_periodic>) -> f64;
        fn inc(self: Pin<&mut c_loop_all_periodic>) -> bool;

        fn new_i32_vector() -> UniquePtr<CxxVector<i32>>;
        fn new_f64_vector() -> UniquePtr<CxxVector<f64>>;

        fn compute_cell(
            cell: Pin<&mut voronoicell_neighbor>,
            container: Pin<&container_periodic>,
            voro_loop: Pin<&c_loop_all_periodic>,
        );
    }
}

const OPTIMAL_PARTICLES: f64 = 5.6;

#[pyclass]
pub struct Voronoi {
    boxdim: BoxDim,
    neighbor_list: nlist::NeighborList,
    polytopes: Vec<Vec<DVec3>>,
    volumes: Vec<f64>,
}

#[pymethods]
impl Voronoi {
    #[new]
    pub fn py_new(boxdim: BoxDim, points: PyReadonlyArray2<f32>) -> Self {
        let shape = points.shape();
        assert!(shape[1] == 3);
        let points = points.as_slice().unwrap();
        // SAFETY: reinterpret numpy array slice into slice Vec3
        let ptr = points.as_ptr() as *const Vec3;
        let points = unsafe { std::slice::from_raw_parts(ptr, shape[0]) };
        Self::new(boxdim, points)
    }
}

impl Voronoi {
    pub fn new(boxdim: BoxDim, points: impl AsRef<[Vec3]>) -> Self {
        let points = points.as_ref();
        let v1 = boxdim.lattice_vector(0);
        let v2 = boxdim.lattice_vector(1);
        let v3 = if boxdim.is_2d() {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            boxdim.lattice_vector(2)
        };

        let block_scale = (points.len() as f64 / OPTIMAL_PARTICLES).powf(1.0 / 3.0);
        let l = boxdim.l();
        let nx = (l.x as f64 * block_scale + 1.0) as i32;
        let ny = (l.y as f64 * block_scale + 1.0) as i32;
        let nz = (l.z as f64 * block_scale + 1.0) as i32;

        let mut container = ffi::new_container_periodic(
            v1.x as f64,
            v2.x as f64,
            v2.y as f64,
            v3.x as f64,
            v3.y as f64,
            v3.z as f64,
            nx,
            ny,
            nz,
            3,
        );
        
        let mut container = container.pin_mut();
        for (i, p) in points.iter().enumerate() {
            container
                .as_mut()
                .put(i as i32, p.x as f64, p.y as f64, p.z as f64);
        }
        
        let mut polytopes = Vec::with_capacity(points.len());
        let mut volumes = Vec::with_capacity(points.len());

        let mut cell = ffi::new_voronoicell_neighbor();
        let mut cell = cell.pin_mut();
        let mut voro_loop = ffi::new_c_loop_all_periodic(container.as_ref());
        let mut voro_loop = voro_loop.pin_mut();
        let mut face_areas = ffi::new_f64_vector();
        let mut face_areas = face_areas.pin_mut();
        let mut face_vertices = ffi::new_i32_vector();
        let mut face_vertices = face_vertices.pin_mut();
        let mut neighbors = ffi::new_i32_vector();
        let mut neighbors = neighbors.pin_mut();
        let mut normals = ffi::new_f64_vector();
        let mut normals = normals.pin_mut();
        let mut vertices = ffi::new_f64_vector();
        let mut vertices = vertices.pin_mut();

        let mut bonds: Vec<nlist::NeighborBond> = Vec::new();
        let mut relative_vertices: Vec<DVec3> = Vec::new();

        if voro_loop.as_mut().start() {
            loop {
                relative_vertices.clear();
                let mut system_vertices: Vec<DVec3> = Vec::new();

                container
                    .as_mut()
                    .compute_cell(cell.as_mut(), voro_loop.as_ref());

                let query_point_id = voro_loop.as_mut().pid();
                let query_point = DVec3::new(
                    voro_loop.as_mut().x(),
                    voro_loop.as_mut().y(),
                    voro_loop.as_mut().z(),
                );

                cell.as_mut().face_areas(face_areas.as_mut());
                cell.as_mut().face_vertices(face_vertices.as_mut());
                cell.as_mut().neighbors(neighbors.as_mut());
                cell.as_mut().normals(normals.as_mut());
                cell.as_mut().vertices(
                    query_point.x,
                    query_point.y,
                    query_point.z,
                    vertices.as_mut(),
                );

                assert!(vertices.len() % 3 == 0);
                for v in vertices.as_slice().chunks_exact(3) {
                    let z = v[2];
                    let z = if boxdim.is_2d() && z >= 0.0 {
                        0.0
                    } else {
                        v[2]
                    };
                    relative_vertices.push(DVec3::new(v[0], v[1], z) - query_point);
                }

                if boxdim.is_2d() {
                    relative_vertices.sort_unstable_by(|a, b| {
                        a.y.atan2(a.x).partial_cmp(&b.y.atan2(b.x)).unwrap()
                    });
                }

                let query_point_system_coords = points[query_point_id as usize].as_dvec3();
                for v in relative_vertices.iter() {
                    system_vertices.push(*v + query_point_system_coords)
                }
                polytopes.push(system_vertices);

                volumes.push(cell.as_mut().volume());

                let normals = normals.as_slice();
                let neighbors = neighbors.as_slice();
                let face_areas = face_areas.as_slice();
                assert!(normals.len() % 3 == 0);
                assert!(neighbors.len() * 3 == normals.len());
                assert!(face_areas.len() == neighbors.len());

                for ((point_id, normal), face_area) in neighbors
                    .iter()
                    .zip(normals.chunks_exact(3))
                    .zip(face_areas)
                {
                    let normal = DVec3::from_slice(normal);
                    if (boxdim.is_2d() && normal.z.abs() > 0.5) || normal == DVec3::ZERO {
                        continue;
                    }

                    let weight = *face_area as f32;
                    let point_system_coords = points[*point_id as usize];

                    let rij =
                        boxdim.wrap(&(point_system_coords - query_point_system_coords.as_vec3()));
                    let distance = rij.length();

                    bonds.push(nlist::NeighborBond::new_weighted(
                        query_point_id as u32,
                        *point_id as u32,
                        distance,
                        weight,
                    ));
                }

                if !voro_loop.as_mut().inc() {
                    break;
                }
            }
        }

        bonds.sort_unstable_by(|a, b| a.partial_cmp_id_ref_weight(b).unwrap());

        let mut query_point_indices = Vec::new();
        let mut point_indices = Vec::new();
        let mut counts = Vec::new();
        let mut segments = Vec::new();
        let mut distances = Vec::new();
        let mut weights = Vec::new();

        let n_points = points.len();
        let mut cur_idx = 0;
        let mut cur_count = 0;
        let mut cur_segment = 0;
        for bond in bonds {
            let idx = bond.query_point_idx;
            while idx != cur_idx {
                counts.push(cur_count);
                segments.push(cur_segment);
                cur_idx += 1;
                cur_segment += cur_count;
                cur_count = 0;
            }

            query_point_indices.push(idx);
            point_indices.push(bond.point_idx);
            distances.push(bond.distance);
            weights.push(bond.weight);

            cur_count += 1;
        }

        while cur_idx < n_points as u32 {
            counts.push(cur_count);
            segments.push(cur_segment);
            cur_idx += 1;
            cur_segment += cur_count;
            cur_count = 0;
        }

        let neighbor_list = nlist::NeighborList {
            query_point_indices,
            point_indices,
            counts,
            segments,
            distances,
            weights,
        };

        Self {
            boxdim,
            neighbor_list,
            polytopes,
            volumes,
        }
    }

    fn boxdim(&self) -> &BoxDim {
        &self.boxdim
    }

    fn neighbor_list(&self) -> &nlist::NeighborList {
        &self.neighbor_list
    }

    fn polytopes(&self) -> &Vec<Vec<DVec3>> {
        &self.polytopes
    }

    fn volumes(&self) -> &Vec<f64> {
        &self.volumes
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_voro() {
        let boxdim = crate::boxdim::BoxDim::cube(10.0);

        // random ndarray of shape (n_points, 3)
        let points = ndarray::Array2::<f32>::random((50000, 3), Uniform::new(-4.0, 4.0));
        let shape = points.shape();
        let points = points.as_slice().unwrap();
        // SAFETY: reinterpret numpy array slice into slice Vec3
        let ptr = points.as_ptr() as *const Vec3;
        let points = unsafe { std::slice::from_raw_parts(ptr, shape[0]) };

        let _voro = Voronoi::new(boxdim, points);
    }
}

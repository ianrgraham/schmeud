use glam::DVec2;
use rand::prelude::*;
use robust::{incircle, Coord};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hasher, Hash};
use ahash::AHasher;
use std::cell::Cell;

use crate::boxdim::BoxDim;

type Point = Coord<f64>;

#[derive(Debug)]
struct BoundBox {
    min: DVec2,
    max: DVec2,
}

impl BoundBox {
    fn new(min: DVec2, max: DVec2) -> Self {
        assert!(min.x <= max.x);
        assert!(min.y <= max.y);
        Self { min, max }
    }
}

enum Orient {
    In,
    Out,
    On,
}

/// Triangle element
struct Triel {
    verts: [usize; 3],
    daught: [Option<usize>; 3],
    live: bool,
}

impl Triel {
    fn new(verts: [usize; 3]) -> Self {
        Self {
            verts,
            daught: [None, None, None],
            live: true,
        }
    }

    fn orient(&self, point: Point, points: &[Point]) -> Orient {
        let mut d: f64;
        let mut on = false;
        for i in 0..3 {
            let j = (i + 1) % 3;
            let pi = points[self.verts[i]];
            let pj = points[self.verts[j]];
            d = (pj.x - pi.x) * (point.y - pi.y) - (pj.y - pi.y) * (point.x - pi.x);
            if d < 0.0 {
                return Orient::Out;
            }
            if d == 0.0 {
                on = true;
            }
        }
        if on {
            Orient::On
        } else {
            Orient::In
        }
    }
}

// Symmetry: H(a,b) = H(b,a).wrapping_neg()
struct AddHasher {
    data: Cell<u64>,
    sign: Cell<u64>,
    inner: DefaultHasher,
}


impl AddHasher {
    fn new() -> Self {
        Self {
            data: Cell::new(0),
            sign: Cell::new(1),
            inner: DefaultHasher::default(),
        }
    }
}

impl Hasher for AddHasher {
    fn write(&mut self, bytes: &[u8]) {
        
        for b in bytes {
            let mut tmp = self.inner.clone();
            tmp.write_u8(*b);
            *self.data.get_mut() += self.sign.get() * tmp.finish();
            self.sign.set(self.sign.get().wrapping_neg());
        }
    }

    fn write_usize(&mut self, i: usize) {
        let mut tmp = self.inner.clone();
        tmp.write_usize(i);
        *self.data.get_mut() += self.sign.get() * tmp.finish();
        self.sign.set(self.sign.get().wrapping_neg());
    }

    fn write_u64(&mut self, i: u64) {
        let mut tmp = self.inner.clone();
        tmp.write_u64(i);
        *self.data.get_mut() ^= self.sign.get() * tmp.finish();
        self.sign.set(self.sign.get().wrapping_neg());
    }

    fn finish(&self) -> u64 {
        let out = self.data.get();
        self.data.set(0);
        self.sign.set(1);
        return out
    }
}

// Symmetry: H(a,b,c) = H(b,c,a) = H(c,a,b) = H(a,c,b) = H(b,a,c) = H(c,b,a)
struct XorHasher {
    data: Cell<u64>,
    inner: AHasher,
}

impl XorHasher {
    fn new() -> Self {
        Self {
            data: Cell::new(0),
            inner: AHasher::default(),
        }
    }
}

impl Hasher for XorHasher {
    fn write(&mut self, bytes: &[u8]) {
        
        for b in bytes {
            let mut tmp = self.inner.clone();
            tmp.write_u8(*b);
            *self.data.get_mut() ^= tmp.finish();
        }
    }

    fn write_usize(&mut self, i: usize) {
        let mut tmp = self.inner.clone();
        tmp.write_usize(i);
        *self.data.get_mut() ^= tmp.finish();
    }

    fn write_u64(&mut self, i: u64) {
        let mut tmp = self.inner.clone();
        tmp.write_u64(i);
        *self.data.get_mut() ^= tmp.finish();
    }

    fn finish(&self) -> u64 {
        let out = self.data.get();
        self.data.set(0);
        return out
    }
}

#[derive(Copy, Clone, Debug)]
struct CcwTri {
    a: usize,
    b: usize,
    c: usize,
}

impl CcwTri {

    fn rotate_inplace(&mut self) {
        let tmp = self.a;
        self.a = self.b;
        self.b = self.c;
        self.c = tmp;
    }

    fn direct_eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }

    fn direct_sum(&self) -> usize {
        self.a + self.b + self.c
    }
}

impl PartialEq for CcwTri {
    fn eq(&self, other: &Self) -> bool {
        // optional quick escape hatch in case they are not equal
        // if self.direct_sum() != other.direct_sum() {
        //     return false;
        // }
        let mut other = other.clone();
        for _ in 0..3 {
            if self.direct_eq(&other) {
                return true;
            }
            other.rotate_inplace();
        }
        false
    }
}

impl Eq for CcwTri {}

impl Hash for CcwTri {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.a);
        state.write_usize(self.b);
        state.write_usize(self.c);
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Line {
    a: usize,
    b: usize,
}

impl Line {
    fn new(a: usize, b: usize) -> Self {
        Self { a, b }
    }
}

impl Hash for Line {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.a);
        state.write_usize(self.b);
    }
}

// TODO the hashmaps could possibly have collision, need a fallback strategy
// FIX define new types Line and CcwTri which implement there own hash and
// equality traits
struct Delaunay {
    points: Vec<Point>,
    triels: Vec<Triel>,
    tree_max: usize,
    line_map: HashMap<u64, usize>,
    triel_map: HashMap<u64, usize>,
    boxdim: Option<BoxDim>,
    hash: ahash::AHasher,
}

impl Delaunay {
    const FUZZ: f64 = 1e-6;
    const BIGSCALE: f64 = 1000.0;

    pub fn new_boxless(mut points: Vec<Point>) -> Self {
        let triels = Vec::with_capacity(2 * points.len() + 6); // lets try this out,
        let tree_max = 10 * points.len() + 1000;
        let line_map = HashMap::with_capacity(6 * points.len() + 12);
        let triel_map = HashMap::with_capacity(2 * points.len() + 6);
        let boxdim = None;

        // make indices and shuffle them
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(&mut thread_rng());

        points.push(Point {
            x: -Delaunay::BIGSCALE,
            y: -Delaunay::BIGSCALE,
        });
        points.push(Point {
            x: Delaunay::BIGSCALE,
            y: -Delaunay::BIGSCALE,
        });
        points.push(Point {
            x: 0.0,
            y: Delaunay::BIGSCALE,
        });

        let vert_indices = [points.len() - 3, points.len() - 2, points.len() - 1];

        let mut delaunay = Self {
            points,
            triels,
            tree_max,
            line_map,
            triel_map,
            boxdim,
            hash: ahash::AHasher::default(),
        };

        delaunay.store_triangle(vert_indices);

        for idx in &vert_indices {
            delaunay.insert_point_idx(*idx);
        }

        let npts = vert_indices.len();
        delaunay.cleanup(npts);

        delaunay
    }

    fn insert_point_idx(&mut self, idx: usize) {
        // TODO can optionally implement fuzzing logic
        let mut legalize = smallvec::SmallVec::<[[usize; 3]; 6]>::new();
        let triel_idx = self
            .which_triel_contains_point(self.points[idx])
            .expect("No existing triel contains the point");
        let triel = &self.triels[triel_idx];
        let v = triel.verts;
        let d0 = self.store_triangle([idx, v[0], v[1]]);
        legalize.push([idx, v[0], v[1]]);
        let d1 = self.store_triangle([idx, v[1], v[2]]);
        legalize.push([idx, v[1], v[2]]);
        let d2 = self.store_triangle([idx, v[2], v[0]]);
        legalize.push([idx, v[2], v[0]]);
        let d = [Some(d0), Some(d1), Some(d2)];

        self.erase_triangle(v, d);

        while let Some(tri) = legalize.pop() {
            let key = Self::line_hash(&self.hash, [tri[1], tri[2]]);
            let Some(&l) = self.line_map.get(&key) else {
                continue;
            };

            let pts = &self.points;
            if incircle(pts[l], pts[2], pts[0], pts[1]) > 0.0 {
                let d0 = self.store_triangle([tri[0], l, tri[2]]);
                legalize.push([tri[0], l, tri[2]]);
                let d1 = self.store_triangle([tri[0], tri[1], l]);
                legalize.push([tri[0], tri[1], l]);
                let d = [Some(d0), Some(d1), None];
                self.erase_triangle(tri, d);
                self.erase_triangle([l, tri[2], tri[1]], d);
                self.line_map.remove(&key);
                self.line_map.remove(&(0u64.wrapping_sub(key)));
            }
        }
    }

    fn which_triel_contains_point(&self, point: Point) -> Option<usize> {
        let mut k = 0;
        while self.triels[k].live {
            let triel = &self.triels[k];
            let mut j;
            for i in 0..3 {
                j = triel.daught[i];
                if let Some(j) = j {
                    let triel = &self.triels[j];
                    match triel.orient(point, &self.points) {
                        Orient::In | Orient::On => {
                            k = j;
                            break;
                        }
                        Orient::Out => {
                            return None;
                        }
                    }
                }
            }
        }
        Some(k)
    }

    fn erase_triangle(&mut self, v: [usize; 3], d: [Option<usize>; 3]) {
        let key = Self::tri_hash(&self.hash, v);
        let triel_idx = self.triel_map.remove(&key).expect("Triangle not found");
        self.triels[triel_idx].live = false;
        self.triels[triel_idx].daught = d;
    }

    fn store_triangle(&mut self, verts: [usize; 3]) -> usize {
        let triel_idx = self.triels.len();
        if triel_idx >= self.tree_max {
            panic!("Tree max exceeded")
        }
        let triel = Triel::new(verts);
        self.triels.push(triel);
        if let Some(_) = self
            .triel_map
            .insert(Self::tri_hash(&self.hash, verts), triel_idx)
        {
            panic!("Triangle already exists")
        }
        for i in 0..3 {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            if let Some(_) = self
                .line_map
                .insert(Self::line_hash(&self.hash, [verts[i], verts[j]]), verts[k])
            {
                panic!("Line already exists")
            }
        }
        triel_idx
    }

    fn cleanup(&mut self, ub: usize) {
        for triel in self.triels.as_mut_slice() {
            if triel.live {
                if triel.verts[0] >= ub || triel.verts[1] >= ub || triel.verts[2] >= ub {
                    triel.live = false;
                    // let h = Self::tri_hash(&self.hash, triel.verts);
                    // self.triel_map.remove(&h).expect("Triangle not found");
                }
            }
        }
    }

    fn tri_hash(h: &ahash::AHasher, v: [usize; 3]) -> u64 {
        let mut ha = h.clone();
        ha.write_usize(v[0]);
        let mut hb = h.clone();
        hb.write_usize(v[1]);
        let mut hc = h.clone();
        hc.write_usize(v[2]);
        ha.finish() ^ hb.finish() ^ hc.finish()
    }

    fn line_hash(h: &ahash::AHasher, v: [usize; 2]) -> u64 {
        let mut hb = h.clone();
        hb.write_usize(v[0]);
        let mut hc = h.clone();
        hc.write_usize(v[1]);
        hb.finish() - hc.finish()
    }
}

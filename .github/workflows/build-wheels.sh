#!/bin/bash
set -e -x

for PYBIN in /opt/python/cp3[89]*/bin; do
    "${PYBIN}/pip" install maturin
    "${PYBIN}/pip" list
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done

cp -r wheelhouse dist
#!/bin/bash
set -e -x

for PYBIN in /opt/python/cp3[7891]*/bin; do
    "${PYBIN}/pip" install maturin>=0.12.15
    "${PYBIN}/pip" list
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done

cp -r wheelhouse dist
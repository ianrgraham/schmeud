#!/bin/bash
set -e -x

for PYBIN in /opt/python/cp3[78910]*/bin; do
    "${PYBIN}/pip" install maturin
    "${PYBIN}/maturin" build -i "${PYBIN}/python" --release
done

for wheel in target/wheels/*.whl; do
    auditwheel repair "${wheel}"
done

mv wheelhouse dist
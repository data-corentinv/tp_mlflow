for entry_point in load features validate train future predict
do
    sed -n '/import/p' foodcast/.application__CORRECTION/$entry_point.py > foodcast/application/$entry_point.py
    rm -f tests/unit_tests/application/test_$entry_point.py
done

./to_conda.sh
make check
make tests

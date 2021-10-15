if [ $# -eq 0 ]; then
    echo please add argument load, features, validate, train, future or predict
else
    cp tests/unit_tests/.application__CORRECTION/test_$1.py tests/unit_tests/application/
fi

./to_conda.sh
make check
make tests

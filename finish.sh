if [ $# -eq 0 ]; then
    cp foodcast/.application__CORRECTION/* foodcast/application/
    cp tests/unit_tests/.application__CORRECTION/* tests/unit_tests/application/
    make lint
else
    cp foodcast/.application__CORRECTION/$1.py foodcast/application/
    cp tests/unit_tests/.application__CORRECTION/test_$1.py tests/unit_tests/application/
fi

./to_conda.sh
make check
make tests

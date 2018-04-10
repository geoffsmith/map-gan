LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64 pipenv run python -m mapgan.gan --job-dir=`pwd`/jobs/ --train-data=./data/train.tfrecords "$@"

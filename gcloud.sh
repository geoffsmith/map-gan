TRAINER_PACKAGE_PATH="/Users/geoff/projects/dcgan-tf/mapgan/"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="map_gan_$now"
MAIN_TRAINER_MODULE="mapgan.gan"
JOB_DIR="gs://map-gan/job"
PACKAGE_STAGING_LOCATION="gs://map-gan/package/"
REGION="europe-west1"
RUNTIME_VERSION="1.6"
gcloud ml-engine jobs submit training $JOB_NAME --package-path=$TRAINER_PACKAGE_PATH --module-name $MAIN_TRAINER_MODULE --job-dir $JOB_DIR --region $REGION --runtime-version 1.6 --python-version 3.5 --scale-tier basic-gpu
# gcloud ml-engine jobs stream-logs $JOB_NAME

#!/usr/bin/env bash
cd $(dirname $0)
version=$(cat ../nermodel/version.py | grep '__version__ = ' | sed "s#__version__ = ##g" | sed "s#'##g")
img_name=nerservice
cd ..
docker build ${EXTRA_BUILD_ARGS} . -t ${img_name}:${version} -t ${img_name}:latest

cat <<EOF
# You can use following commands to push images.
docker push ${img_name}:${version}
docker push ${img_name}:latest
EOF
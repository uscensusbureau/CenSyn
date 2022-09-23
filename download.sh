#!/bin/bash
# Goes through all the requirements and downloads them for a specific OS

set -e

if [[ -z $1 ]]; then
    echo "Must provide a wheelhouse directory."
    exit 1
fi
if [[ -z $2 ]]; then
    echo "Must provide the desired platform."
    exit 1
fi

WHEELHOUSE_DIR="$1"
PLATFORM="$2"

# Create the wheelhouse dir if needed.
[[ -d "$WHEELHOUSE_DIR" ]] || mkdir "$WHEELHOUSE_DIR"

# Download as generally as possible.
dl () {
    echo "Downloading $1..."
    pip download --only-binary=:all: --platform any -d ${WHEELHOUSE_DIR} $1 ||
    pip download --only-binary=:all: --platform ${PLATFORM} -d ${WHEELHOUSE_DIR} $1 ||
    pip download -d ${WHEELHOUSE_DIR} $1
}

# Indicate the failed dependencies.
fail_count=0
failed=()

# Go through all the requirements and download them in the most general way possible.
while read -r line; do
    if dl ${line}; then
        true
    else
        failed+=(${line})
        fail_count=$(($fail_count+1))
    fi
done < requirements.txt

# Also download wheel so that wheeling can be done.
dl wheel

# Output any failed packages.
if [[ ${fail_count} -gt 0 ]]; then
    echo -n "$fail_count requirements failed: "; echo ${failed[@]}
fi

# Exit with a useful status.
exit ${fail_count}

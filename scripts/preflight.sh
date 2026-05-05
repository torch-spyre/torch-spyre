#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="oc_install.log"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

log "Starting OpenShift CLI check"

if [ -x /usr/local/bin/oc ]; then
    log "oc already installed"
    /usr/local/bin/oc version --client | tee -a "$LOG_FILE"
    exit 0
fi

log "oc not found. Installing..."

TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

log "Downloading oc client"
curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/openshift-client-linux.tar.gz

log "Extracting archive"
tar -xvf openshift-client-linux.tar.gz

log "Installing binary"
sudo mv oc /usr/local/bin/

log "Validating installation"
/usr/local/bin/oc version --client | tee -a "$LOG_FILE"

log "Installation completed"